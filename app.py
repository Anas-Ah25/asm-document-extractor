from __future__ import annotations
import json
import copy
import time
import re
import io
import zipfile
from pathlib import Path
from typing import Any

import streamlit as st
import pandas as pd

from extractor import (
    parse_document, full_explore,
    extract_schema_regex, extract_schema_ollama,
    infer_schema,
    DEFAULT_SCHEMA,
)

POOL_DIR = Path(__file__).parent
SCHEMA_FILE = POOL_DIR / "schema.json"
SUPPORTED = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

# Connection credentials
_s = st.secrets if hasattr(st, "secrets") else {}
LLM_BASE_URL: str = _s.get("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY: str = _s.get("API_KEY", "")
MODEL_NAME: str = _s.get("MODEL_NAME", "gemma4:31b-cloud")

# Page Layout
st.set_page_config(page_title="Document Extractor", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
.stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
div[data-testid="stFileUploaderDropzone"] { padding: 0.6rem; }
</style>
""", unsafe_allow_html=True)

# Names that identify the OLD engineering schema — if found, auto-migrate
_OLD_SCHEMA_MARKER = "document_number"

# Schema persistence helpers
def load_schema() -> list[dict]:
    if SCHEMA_FILE.exists():
        try:
            with open(SCHEMA_FILE, "r") as f:
                data = json.load(f)
            # Auto-migrate: if this is the old engineering schema, discard it
            if any(field.get("name") == _OLD_SCHEMA_MARKER for field in data):
                SCHEMA_FILE.unlink(missing_ok=True)
            # Sanitize loaded schema to ensure fields are populated
            for f in data:
                if isinstance(f, dict):
                    f.setdefault("enabled", True)
                    f.setdefault("description", "")
                    f.setdefault("field_type", "scalar")
                    if f.get("field_type") == "table" and "columns" in f:
                        for col in f["columns"]:
                            if isinstance(col, dict):
                                col.setdefault("enabled", True)
                                col.setdefault("description", "")
            return data
        except Exception as e:
            st.sidebar.error(f"Error loading schema.json: {e}")
    return copy.deepcopy(DEFAULT_SCHEMA)

def save_schema(schema_data: list[dict]) -> None:
    try:
        with open(SCHEMA_FILE, "w") as f:
            json.dump(schema_data, f, indent=2)
    except Exception as e:
        st.sidebar.error(f"Error saving schema: {e}")

# Session States
_defaults = {
    "schema": None,
    "results_cache": {},
    "explore_result": None,
    "explore_parsed": None,
    "single_result": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = load_schema() if k == "schema" else v

# Auto-migrate stale session: if session has old engineering schema, replace it
if any(f.get("name") == _OLD_SCHEMA_MARKER for f in (st.session_state.schema or [])):
    st.session_state.schema = copy.deepcopy(DEFAULT_SCHEMA)
    save_schema(st.session_state.schema)

RESERVED_STEMS = {"app", "extractor", "clean_app"}

# File listing helper
def _disk_files() -> list[Path]:
    return sorted(
        [f for f in POOL_DIR.iterdir()
         if f.suffix.lower() in SUPPORTED and f.stem not in RESERVED_STEMS],
        key=lambda f: f.name,
    )

def _all_names() -> list[str]:
    return [f.name for f in _disk_files()]

def _size_kb(name: str) -> int:
    p = POOL_DIR / name
    return p.stat().st_size // 1024 if p.exists() else 0

# Helper to clean legacy wrapped citation values from cache
def unwrap_citations(data: Any) -> Any:
    if isinstance(data, dict):
        if "value" in data and "citation" in data:
            return unwrap_citations(data["value"])
        return {k: unwrap_citations(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [unwrap_citations(x) for x in data]
    return data

# Duplicate header normalization
def _dedup(headers: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for h in headers:
        h = str(h).strip() or "col"
        seen[h] = seen.get(h, 0) + 1
        out.append(h if seen[h] == 1 else f"{h}_{seen[h]}")
    return out

# Cel formatting and truncation
def _cell(val: object, max_len: int = 120) -> str:
    s = re.sub(r"\s*\n\s*", " · ", str(val or "").strip())
    s = re.sub(r"  +", " ", s)
    return s[:max_len] + "…" if len(s) > max_len else s

# Clean table printing helper
def _render_tbl(rows: list[list]) -> None:
    if not rows:
        st.caption("(empty)")
        return
    hdr = [str(c or "").strip() for c in rows[0]]
    data = rows[1:]
    if hdr.count("") / max(len(hdr), 1) > 0.5:
        hdr, data = [f"col_{i}" for i in range(len(hdr))], rows
    hdr = _dedup(hdr)
    cleaned = []
    for r in data:
        row = [_cell(c) for c in r]
        while len(row) < len(hdr):
            row.append("")
        cleaned.append(row[:len(hdr)])
    if not cleaned:
        st.caption("(no data rows)")
        return
    df = pd.DataFrame(cleaned, columns=hdr)
    df = df.loc[:, (df != "").any(axis=0)]
    st.dataframe(df, use_container_width=True, hide_index=True)

# Sidebar configurations
with st.sidebar:
    st.title("Settings")

    parser_choice = st.selectbox(
        "Parser",
        ["OCR supported", "Fast"],
        help="OCR supported utilizes layout-aware parsing. Fast uses standard extraction.",
    )
    parser_key = "pymupdf4llm" if parser_choice == "OCR supported" else "pymupdf"

    use_llm = True

    st.divider()
    st.subheader("Target Schema")

    schema = st.session_state.schema

    editor_mode = st.segmented_control(
        "Edit Mode",
        ["UI Builder", "JSON Editor"],
        default="UI Builder",
        key="schema_edit_mode",
    )

    if editor_mode == "JSON Editor":
        schema_json = json.dumps(schema, indent=2)
        schema_help = """Edit the schema directly as a JSON list.

### How to construct a valid JSON schema:
1. **Root element** must be a JSON array of objects: `[ { ... } ]`.
2. **Every field object** must contain:
   - `"name"`: Unique string identifier (e.g., `"invoice_date"`). Use snake_case.
   - `"field_type"`: Must be either `"scalar"` (for single values) or `"table"` (for rows of items).
   - `"description"`: Text explanation guiding the extraction model.
3. **Table fields** must also include:
   - `"columns"`: A JSON array of column objects, where each object has `"name"` and `"description"`.
4. **Primary key** (optional): Add `"primary_key": true` to a scalar field to include it as a linking column in table CSVs.

### Example:
```json
[
  {
    "name": "invoice_no",
    "field_type": "scalar",
    "description": "Invoice number",
    "primary_key": true
  },
  {
    "name": "line_items",
    "field_type": "table",
    "description": "Items listed",
    "columns": [
      { "name": "desc", "description": "Description" },
      { "name": "qty", "description": "Quantity" }
    ]
  }
]
```"""
        with st.form("json_schema_form"):
            json_input = st.text_area(
                "Schema JSON",
                value=schema_json,
                height=350,
                help=schema_help,
            )
            submitted = st.form_submit_button("Apply JSON Schema", use_container_width=True)
            if submitted:
                try:
                    parsed = json.loads(json_input)
                    if not isinstance(parsed, list):
                        st.error("Schema must be a JSON list of dictionaries.")
                    else:
                        # Sanitize schema and default omitted 'enabled' flag to True
                        for f in parsed:
                            if isinstance(f, dict):
                                f.setdefault("enabled", True)
                                f.setdefault("description", "")
                                f.setdefault("field_type", "scalar")
                                if f.get("field_type") == "table" and "columns" in f:
                                    for col in f["columns"]:
                                        if isinstance(col, dict):
                                            col.setdefault("enabled", True)
                                            col.setdefault("description", "")
                        st.session_state.schema = parsed
                        save_schema(parsed)
                        st.success("Schema updated successfully!")
                        st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
    else:
        # Scalar configurations UI
        with st.expander("Scalar fields", expanded=True):
            for f in [f for f in schema if f["field_type"] == "scalar"]:
                idx = schema.index(f)
                c1, c2, c3, c4 = st.columns([1, 4, 1, 1])
                schema[idx]["enabled"] = c1.checkbox(
                    f["name"], value=f["enabled"], key=f"sf_{f['name']}",
                    label_visibility="collapsed",
                )
                c2.caption(f"**{f['name']}** — {f['description'][:48]}")
                schema[idx]["primary_key"] = c3.checkbox(
                    "🔑", value=f.get("primary_key", False), key=f"pk_{f['name']}",
                    help="Include as primary key in table CSVs",
                )
                if c4.button("✕", key=f"rm_sf_{f['name']}", help="Remove field"):
                    schema.pop(idx)
                    save_schema(schema)
                    st.rerun()

            with st.form("add_scalar", clear_on_submit=True):
                c1, c2 = st.columns(2)
                sname = c1.text_input("Name", placeholder="po_number")
                sdesc = c2.text_input("Description", placeholder="Purchase order #")
                if st.form_submit_button("+ Add field", use_container_width=True) and sname.strip():
                    normalized_name = sname.strip().lower().replace(" ", "_")
                    if any(f["name"] == normalized_name for f in schema):
                        st.error(f"Field '{normalized_name}' already exists.")
                    else:
                        schema.append({
                            "name": normalized_name,
                            "description": sdesc.strip() or sname.strip(),
                            "field_type": "scalar", "enabled": True,
                        })
                        save_schema(schema)
                        st.rerun()

        # Table configurations UI
        with st.expander("Table fields", expanded=True):
            for f in [f for f in schema if f["field_type"] == "table"]:
                idx = schema.index(f)
                c1, c2, c3 = st.columns([1, 5, 1])
                schema[idx]["enabled"] = c1.checkbox(
                    f["name"], value=f["enabled"], key=f"tf_{f['name']}",
                    label_visibility="collapsed",
                )
                c2.caption(f"**{f['name']}** — {f['description'][:38]}")
                if c3.button("✕", key=f"rm_tf_{f['name']}", help="Remove table field"):
                    schema.pop(idx)
                    save_schema(schema)
                    st.rerun()

                if not f["enabled"]:
                    continue

                cur_mode = f.get("columns_mode", "defined")
                new_mode = st.segmented_control(
                    "Columns",
                    ["Auto-detect", "Defined"],
                    default="Auto-detect" if cur_mode == "auto" else "Defined",
                    key=f"cmode_{f['name']}",
                )
                schema[idx]["columns_mode"] = "auto" if new_mode == "Auto-detect" else "defined"

                if schema[idx]["columns_mode"] == "auto":
                    st.caption("The model will automatically infer all columns.")
                else:
                    cols_list = schema[idx].get("columns", [])
                    for ci, col in enumerate(cols_list):
                        cc1, cc2, cc3 = st.columns([1, 5, 1])
                        schema[idx]["columns"][ci]["enabled"] = cc1.checkbox(
                            col["name"], value=col["enabled"],
                            key=f"col_{f['name']}_{col['name']}",
                            label_visibility="collapsed",
                        )
                        cc2.caption(f"`{col['name']}` — {col['description'][:32]}")
                        if cc3.button("✕", key=f"rm_col_{f['name']}_{ci}", help="Remove column"):
                            schema[idx]["columns"].pop(ci)
                            save_schema(schema)
                            st.rerun()

                    with st.form(f"add_col_{f['name']}", clear_on_submit=True):
                        ac1, ac2 = st.columns(2)
                        cname = ac1.text_input("Column name", placeholder="item_number", key=f"cn_{f['name']}")
                        cdesc = ac2.text_input("Description",  placeholder="Item #",      key=f"cd_{f['name']}")
                        if st.form_submit_button("+ Add column", use_container_width=True) and cname.strip():
                            normalized_name = cname.strip().lower().replace(" ", "_")
                            existing_cols = schema[idx].get("columns", [])
                            if any(c["name"] == normalized_name for c in existing_cols):
                                st.error(f"Column '{normalized_name}' already exists in this table.")
                            else:
                                schema[idx]["columns"].append({
                                    "name": normalized_name,
                                    "description": cdesc.strip() or cname.strip(),
                                    "enabled": True,
                                })
                                save_schema(schema)
                                st.rerun()

                st.divider()

            with st.form("add_table", clear_on_submit=True):
                tname = st.text_input("Table name", placeholder="line_items")
                tdesc = st.text_input("Description", placeholder="Order line items")
                tauto = st.toggle("Auto-detect columns", value=True,
                                  help="Allows dynamic mapping of tables without fixed schemas.")
                if st.form_submit_button("+ Add table field", use_container_width=True) and tname.strip():
                    normalized_name = tname.strip().lower().replace(" ", "_")
                    if any(f["name"] == normalized_name for f in schema):
                        st.error(f"Field '{normalized_name}' already exists.")
                    else:
                        schema.append({
                            "name": normalized_name,
                            "description": tdesc.strip() or tname.strip(),
                            "field_type": "table", "enabled": True,
                            "columns_mode": "auto" if tauto else "defined",
                            "columns": [],
                        })
                        save_schema(schema)
                        st.rerun()

        c1, c2 = st.columns(2)
        if c1.button("Reset schema", use_container_width=True):
            st.session_state.schema = copy.deepcopy(DEFAULT_SCHEMA)
            save_schema(st.session_state.schema)
            st.rerun()
        if c2.button("Infer schema", use_container_width=True,
                     help="Let the LLM suggest a schema from a sample document"):
            st.session_state._infer_pending = True

        if st.session_state.get("_infer_pending"):
            names_now = _all_names()
            if names_now:
                pick = st.selectbox("Sample document", names_now, key="infer_pick")
                if st.button("Run", key="run_infer", use_container_width=True):
                    path = POOL_DIR / pick
                    if path.exists():
                        with st.spinner("Inferring schema…"):
                            parsed = parse_document(path, parser=parser_key,
                                                     llm_base_url=LLM_BASE_URL, api_key=API_KEY, model=MODEL_NAME)
                            suggested = infer_schema(parsed, LLM_BASE_URL, API_KEY, MODEL_NAME)
                        if suggested:
                            st.session_state.schema = suggested
                            save_schema(suggested)
                            st.session_state._infer_pending = False
                            st.rerun()
            else:
                st.caption("Upload a file first.")
                st.session_state._infer_pending = False

    # Auto-save schema at the end of sidebar configuration block
    save_schema(schema)

# Run extraction logic helper
def _run_extraction(name: str) -> tuple[dict, float]:
    path = POOL_DIR / name
    if not path.exists():
        raise FileNotFoundError(name)
    parsed = parse_document(path, parser=parser_key,
                            llm_base_url=LLM_BASE_URL, api_key=API_KEY, model=MODEL_NAME)
    if use_llm:
        result, elapsed = extract_schema_ollama(
            parsed, st.session_state.schema, LLM_BASE_URL, API_KEY, MODEL_NAME)
    else:
        result, elapsed = extract_schema_regex(parsed, st.session_state.schema)
    return result, elapsed

def _extract_scalars(result: dict) -> dict:
    schema = st.session_state.schema
    row = {}
    for f in schema:
        if not f["enabled"]:
            continue
        name = f["name"]
        val = result.get(name)
        if f["field_type"] == "scalar":
            row[name] = val.get("value", "") if isinstance(val, dict) else (val or "")
        elif f["field_type"] == "table":
            t_rows = _extract_table(result, name)
            row[name] = json.dumps(t_rows, ensure_ascii=False) if t_rows else ""
    return row

def _extract_table(result: dict, table_name: str) -> list[dict]:
    schema = st.session_state.schema
    for f in schema:
        if f["name"] == table_name and f["field_type"] == "table" and f["enabled"]:
            val = result.get(table_name)
            raw_rows = val if isinstance(val, list) else []
            cleaned_rows = []
            for r in raw_rows:
                if isinstance(r, dict):
                    cleaned_row = {}
                    for col_k, col_v in r.items():
                        cleaned_row[col_k] = col_v.get("value", "") if isinstance(col_v, dict) else (col_v or "")
                    cleaned_rows.append(cleaned_row)
                else:
                    cleaned_rows.append({"value": str(r)})
            return cleaned_rows
    return []

# Render details on extraction completion
def _show_result(result: dict, elapsed: float, fname: str) -> None:
    st.success(f"**{fname}** — Success")

    schema = st.session_state.schema
    scalar_names = {f["name"] for f in schema if f["field_type"] == "scalar" and f["enabled"]}
    table_names  = {f["name"] for f in schema if f["field_type"] == "table"  and f["enabled"]}

    scalars = {k: v for k, v in result.items() if k in scalar_names}
    if scalars:
        cols = st.columns(3)
        for ci, (field, wrapped) in enumerate(scalars.items()):
            val  = wrapped.get("value", "") if isinstance(wrapped, dict) else wrapped
            with cols[ci % 3]:
                st.markdown(f"**{field}**")
                st.code(val or "—", language=None)

    for field in table_names:
        rows_raw = result.get(field, [])
        st.markdown(f"**{field}**")
        if not rows_raw:
            st.caption("No rows found.")
            continue
        rows = [{k: _cell(v.get("value", "") if isinstance(v, dict) else v)
                 for k, v in row.items()} for row in rows_raw]
        df = pd.DataFrame(rows)
        df.columns = _dedup(list(df.columns))
        st.dataframe(df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns([1, 4])
    c1.download_button(
        "Download JSON",
        data=json.dumps(unwrap_citations(result), indent=2, default=str),
        file_name=f"{Path(fname).stem}_extracted.json",
        mime="application/json",
        key=f"dl_{fname}_{time.time()}",
    )
    with c2.expander("Raw JSON"):
        st.json(unwrap_citations(result), expanded=1)

# Streamlit App Main Area
st.title("Document Extractor")
tab_extract, tab_explore = st.tabs(["Extract", "Explore"])

with tab_extract:
    # Workspace management
    with st.container(border=True):
        up_col, info_col = st.columns([3, 1])
        with up_col:
            new_uploads = st.file_uploader(
                "Drop files here",
                type=["pdf", "docx", "png", "jpg", "jpeg", "tiff", "bmp"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
        with info_col:
            st.caption("PDF or DOCX  \nFiles are saved permanently.")

        if new_uploads:
            added = 0
            for uf in new_uploads:
                dest = POOL_DIR / uf.name
                if not dest.exists():
                    dest.write_bytes(uf.read())
                    added += 1
            if added:
                st.rerun()

        names = _all_names()
        selected: list[str] = []

        if names:
            st.divider()
            hc1, hc2, hc3, hc4, hc5 = st.columns([4, 1, 1, 1, 1])
            hc1.caption("File"); hc2.caption("Size"); hc3.caption("Status")
            hc4.caption("Select"); hc5.caption("Remove")

            for name in names:
                cached = name in st.session_state.results_cache
                status = "done" if cached else "—"

                fc1, fc2, fc3, fc4, fc5 = st.columns([4, 1, 1, 1, 1])
                fc1.markdown(name)
                fc2.caption(f"{_size_kb(name)} KB")
                fc3.caption(status)
                if fc4.checkbox("select", key=f"sel_{name}", label_visibility="collapsed"):
                    selected.append(name)
                if fc5.button("✕", key=f"rm_{name}", help="Remove"):
                    (POOL_DIR / name).unlink(missing_ok=True)
                    st.session_state.results_cache.pop(name, None)
                    st.rerun()

    # Batch extraction operations
    if names:
        st.write("")
        ac1, ac2, ac3 = st.columns([3, 3, 2])
        n_sel = len(selected)
        run_sel = ac1.button(
            f"Extract selected ({n_sel})" if n_sel else "Extract selected",
            disabled=n_sel == 0, type="primary",
        )
        run_all = ac2.button("Extract all", type="primary")
        clr     = ac3.button("Clear results")

        if clr:
            st.session_state.results_cache = {}
            st.session_state.single_result = None
            st.rerun()

        targets = selected if run_sel else (_all_names() if run_all else [])
        if targets:
            prog = st.progress(0, text="Starting…")
            for i, name in enumerate(targets):
                prog.progress(i / len(targets), text=f"Processing {name}…")
                try:
                    with st.spinner(f"Extracting {name}…"):
                        res, el = _run_extraction(name)
                    st.session_state.results_cache[name] = {"result": res, "elapsed": el}
                except Exception as exc:
                    st.session_state.results_cache[name] = {"error": str(exc)}
            prog.progress(1.0, text="Done")
            st.rerun()

    # Batch results view
    if st.session_state.results_cache:
        st.divider()
        st.subheader("Batch results")

        schema = st.session_state.schema
        table_names = [f["name"] for f in schema if f["field_type"] == "table" and f["enabled"]]

        overview_rows = []
        table_rows_map = {t: [] for t in table_names}

        for fname, data in st.session_state.results_cache.items():
            if "error" in data:
                overview_rows.append({"file": fname, "error": data["error"]})
            else:
                scalars = _extract_scalars(data["result"])
                scalars["file"] = fname
                scalars["time_s"] = f"{data['elapsed']:.1f}s"
                overview_rows.append(scalars)

                for t in table_names:
                    t_rows = _extract_table(data["result"], t)
                    pk_fields = {f["name"] for f in schema if f.get("primary_key") and f["field_type"] == "scalar" and f["enabled"]}
                    for tr in t_rows:
                        combined_row = {"file": fname}
                        for k, v in scalars.items():
                            if k in pk_fields:
                                combined_row[k] = v
                        combined_row.update(tr)
                        table_rows_map[t].append(combined_row)

        df_overview = pd.DataFrame(overview_rows)
        df_overview.columns = _dedup(list(df_overview.columns))
        cols_order_overview = ["file"] + [c for c in df_overview.columns if c != "file"]
        
        tab_titles = ["Overview"] + table_names
        tabs = st.tabs(tab_titles)

        # Pre-build table DataFrames for reuse across tabs and ZIP
        table_dfs = {}
        for t in table_names:
            if table_rows_map[t]:
                df_t = pd.DataFrame(table_rows_map[t])
                df_t.columns = _dedup(list(df_t.columns))
                cols_order_t = ["file"] + [c for c in df_t.columns if c != "file"]
                table_dfs[t] = df_t[cols_order_t]

        batch_json_data = {fn: unwrap_citations(d.get("result", {"error": d.get("error")})) for fn, d in st.session_state.results_cache.items()}

        with tabs[0]:
            st.dataframe(df_overview[cols_order_overview], use_container_width=True, hide_index=True)

            c1, c2, c3 = st.columns(3)
            c1.download_button(
                "Download all (JSON)",
                data=json.dumps(batch_json_data, indent=2, default=str),
                file_name="batch_extracted.json",
                mime="application/json",
                use_container_width=True,
            )
            c2.download_button(
                "Download Overview (CSV)",
                data=df_overview[cols_order_overview].to_csv(index=False),
                file_name="batch_overview.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Build ZIP: overview.csv + batch.json + one CSV per table
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("overview.csv", df_overview[cols_order_overview].to_csv(index=False))
                zf.writestr("batch_extracted.json", json.dumps(batch_json_data, indent=2, default=str))
                for t_name, t_df in table_dfs.items():
                    zf.writestr(f"tables/{t_name.replace(' ', '_')}.csv", t_df.to_csv(index=False))
            zip_buf.seek(0)

            c3.download_button(
                "Download all (ZIP)",
                data=zip_buf.getvalue(),
                file_name="batch_extraction.zip",
                mime="application/zip",
                use_container_width=True,
            )

            with st.expander("Preview Batch JSON"):
                st.json(batch_json_data)

        for i, t in enumerate(table_names):
            with tabs[i+1]:
                if t in table_dfs:
                    st.dataframe(table_dfs[t], use_container_width=True, hide_index=True)
                    st.download_button(
                        f"Download {t} (CSV)",
                        data=table_dfs[t].to_csv(index=False),
                        file_name=f"batch_{t.replace(' ', '_')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    st.info(f"No rows found for table '{t}' across all documents.")

        with st.expander("Per-document detail"):
            for fname, data in st.session_state.results_cache.items():
                st.markdown(f"### {fname}")
                if "error" in data:
                    st.error(data["error"])
                else:
                    _show_result(data["result"], data["elapsed"], fname)

# Explorer View Tab
with tab_explore:
    st.markdown("Parse a document and inspect all its raw content — text, tables, layout.")
    names = _all_names()

    if not names:
        st.info("Upload files in the Extract tab first.")
    else:
        left, right = st.columns([1, 2])
        with left:
            exp_name = st.selectbox("Select file", names, key="exp_sel")
            explore_btn = st.button("Parse & Explore", type="primary", disabled=not exp_name)

        with right:
            if explore_btn and exp_name:
                exp_path = POOL_DIR / exp_name
                if exp_path.exists():
                    with st.spinner(f"Parsing with {parser_key}…"):
                        ep = parse_document(exp_path, parser=parser_key,
                                             llm_base_url=LLM_BASE_URL, api_key=API_KEY, model=MODEL_NAME)
                    st.session_state.explore_result = full_explore(ep)
                    st.session_state.explore_parsed = ep

            er = st.session_state.explore_result
            if er:
                m1, m2, m3 = st.columns(3)
                m1.metric("Characters",   f"{er['total_chars']:,}")
                m2.metric("Text spans",   er["span_elements"])
                m3.metric("Tables found", er["tables_found"])

                t1, t2, t3 = st.tabs(["Full text", "Tables", "Raw JSON"])

                with t1:
                    preview = er["text"][:15000]
                    if len(er["text"]) > 15000:
                        preview += "\n\n…(truncated)"
                    st.markdown(preview)
                    st.download_button("Download text", data=er["text"],
                                       file_name="full_text.md", mime="text/markdown")

                with t2:
                    if not er["tables"]:
                        st.info("No tables detected.")
                    for i, tbl in enumerate(er["tables"]):
                        rows = tbl.get("data", [])
                        with st.expander(
                            f"Table {i+1} — page {tbl['page']}  "
                            f"({tbl['rows']} rows × {tbl['cols']} cols)",
                            expanded=(i == 0),
                        ):
                            _render_tbl(rows)
                            if rows:
                                hdr = _dedup([str(c or "").strip() for c in rows[0]])
                                data_rows = [[_cell(c) for c in r] for r in rows[1:]]
                                buf = io.StringIO()
                                buf.write(",".join(hdr) + "\n")
                                for r in data_rows:
                                    buf.write(",".join(f'"{v}"' for v in r) + "\n")
                                st.download_button(
                                    "Download CSV",
                                    data=buf.getvalue(),
                                    file_name=f"table_{i+1}_p{tbl['page']}.csv",
                                    mime="text/csv",
                                    key=f"csv_{i}",
                                )

                with t3:
                    st.download_button("Download JSON", data=json.dumps(er, indent=2, default=str),
                                       file_name="full_explore.json", mime="application/json")
                    st.json(er, expanded=1)
