from __future__ import annotations
import json
import copy
import time
import re
import io
from pathlib import Path

import streamlit as st
import pandas as pd

from extractor import (
    parse_document, full_explore,
    extract_schema_regex, extract_schema_ollama,
    infer_schema, attach_citations,
    DEFAULT_SCHEMA,
)

POOL_DIR = Path(__file__).parent
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
        st.session_state[k] = copy.deepcopy(DEFAULT_SCHEMA) if k == "schema" else v

RESERVED_STEMS = {"app", "extractor"}

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

    backend_choice = st.radio(
        "Extraction backend",
        [f"LLM support — {MODEL_NAME}", "Regex (fast, no LLM)"],
    )
    use_llm = backend_choice.startswith("LLM")

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
        with st.form("json_schema_form"):
            json_input = st.text_area(
                "Schema JSON",
                value=schema_json,
                height=350,
                help="Edit the schema directly as a JSON list of field definitions.",
            )
            submitted = st.form_submit_button("Apply JSON Schema", use_container_width=True)
            if submitted:
                try:
                    parsed = json.loads(json_input)
                    if not isinstance(parsed, list):
                        st.error("Schema must be a JSON list of dictionaries.")
                    else:
                        st.session_state.schema = parsed
                        st.success("Schema updated successfully!")
                        st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
    else:
        # Scalar configurations UI
        with st.expander("Scalar fields", expanded=True):
            for f in [f for f in schema if f["field_type"] == "scalar"]:
                idx = schema.index(f)
                c1, c2 = st.columns([1, 5])
                schema[idx]["enabled"] = c1.checkbox(
                    f["name"], value=f["enabled"], key=f"sf_{f['name']}",
                    label_visibility="collapsed",
                )
                c2.caption(f"**{f['name']}** — {f['description'][:48]}")

            with st.form("add_scalar", clear_on_submit=True):
                c1, c2 = st.columns(2)
                sname = c1.text_input("Name", placeholder="po_number")
                sdesc = c2.text_input("Description", placeholder="Purchase order #")
                if st.form_submit_button("+ Add field", use_container_width=True) and sname.strip():
                    schema.append({
                        "name": sname.strip().lower().replace(" ", "_"),
                        "description": sdesc.strip() or sname.strip(),
                        "field_type": "scalar", "enabled": True,
                    })
                    st.rerun()

        # Table configurations UI
        with st.expander("Table fields", expanded=True):
            for f in [f for f in schema if f["field_type"] == "table"]:
                idx = schema.index(f)
                c1, c2 = st.columns([1, 5])
                schema[idx]["enabled"] = c1.checkbox(
                    f["name"], value=f["enabled"], key=f"tf_{f['name']}",
                    label_visibility="collapsed",
                )
                c2.caption(f"**{f['name']}** — {f['description'][:38]}")

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
                            st.rerun()

                    with st.form(f"add_col_{f['name']}", clear_on_submit=True):
                        ac1, ac2 = st.columns(2)
                        cname = ac1.text_input("Column name", placeholder="item_number", key=f"cn_{f['name']}")
                        cdesc = ac2.text_input("Description",  placeholder="Item #",      key=f"cd_{f['name']}")
                        if st.form_submit_button("+ Add column", use_container_width=True) and cname.strip():
                            schema[idx]["columns"].append({
                                "name": cname.strip().lower().replace(" ", "_"),
                                "description": cdesc.strip() or cname.strip(),
                                "enabled": True,
                            })
                            st.rerun()

                st.divider()

            with st.form("add_table", clear_on_submit=True):
                tname = st.text_input("Table name", placeholder="line_items")
                tdesc = st.text_input("Description", placeholder="Order line items")
                tauto = st.toggle("Auto-detect columns", value=True,
                                  help="Allows dynamic mapping of tables without fixed schemas.")
                if st.form_submit_button("+ Add table field", use_container_width=True) and tname.strip():
                    schema.append({
                        "name": tname.strip().lower().replace(" ", "_"),
                        "description": tdesc.strip() or tname.strip(),
                        "field_type": "table", "enabled": True,
                        "columns_mode": "auto" if tauto else "defined",
                        "columns": [],
                    })
                    st.rerun()

        c1, c2 = st.columns(2)
        if c1.button("Reset schema", use_container_width=True):
            st.session_state.schema = copy.deepcopy(DEFAULT_SCHEMA)
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
                            st.session_state._infer_pending = False
                            st.rerun()
            else:
                st.caption("Upload a file first.")
                st.session_state._infer_pending = False

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
    return attach_citations(result, parsed["elements"]), elapsed

def _flatten_result(result: dict) -> dict:
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
            raw_rows = val if isinstance(val, list) else []
            cleaned_rows = []
            for r in raw_rows:
                if isinstance(r, dict):
                    cleaned_row = {}
                    for col_k, col_v in r.items():
                        cleaned_row[col_k] = col_v.get("value", "") if isinstance(col_v, dict) else (col_v or "")
                    cleaned_rows.append(cleaned_row)
                else:
                    cleaned_rows.append(r)
            row[name] = json.dumps(cleaned_rows, ensure_ascii=False) if cleaned_rows else ""
    return row

# Render details on extraction completion
def _show_result(result: dict, elapsed: float, fname: str) -> None:
    backend = f"Ollama ({MODEL_NAME})" if use_llm else "Regex"
    st.success(f"**{fname}** — {parser_key} + {backend} — {elapsed:.1f}s")

    schema = st.session_state.schema
    scalar_names = {f["name"] for f in schema if f["field_type"] == "scalar" and f["enabled"]}
    table_names  = {f["name"] for f in schema if f["field_type"] == "table"  and f["enabled"]}

    scalars = {k: v for k, v in result.items() if k in scalar_names}
    if scalars:
        cols = st.columns(3)
        for ci, (field, wrapped) in enumerate(scalars.items()):
            val  = wrapped.get("value", "") if isinstance(wrapped, dict) else wrapped
            cite = wrapped.get("citation")  if isinstance(wrapped, dict) else None
            with cols[ci % 3]:
                st.markdown(f"**{field}**")
                st.code(val or "—", language=None)
                if cite:
                    pg  = cite.get("page")
                    src = cite.get("source_text", "")[:40]
                    st.caption(f"p.{pg} · {src!r}" if pg else src)

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
        data=json.dumps(result, indent=2, default=str),
        file_name=f"{Path(fname).stem}_extracted.json",
        mime="application/json",
        key=f"dl_{fname}_{time.time()}",
    )
    with c2.expander("Raw JSON"):
        st.json(result, expanded=1)

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

        rows = []
        for fname, data in st.session_state.results_cache.items():
            if "error" in data:
                rows.append({"file": fname, "error": data["error"]})
            else:
                r = _flatten_result(data["result"])
                r["file"] = fname
                r["time_s"] = f"{data['elapsed']:.1f}s"
                rows.append(r)

        df = pd.DataFrame(rows)
        df.columns = _dedup(list(df.columns))
        cols_order = ["file"] + [c for c in df.columns if c != "file"]
        st.dataframe(df[cols_order], use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        c1.download_button(
            "Download all (JSON)",
            data=json.dumps(
                {fn: d.get("result", {"error": d.get("error")})
                 for fn, d in st.session_state.results_cache.items()},
                indent=2, default=str,
            ),
            file_name="batch_extracted.json",
            mime="application/json",
            use_container_width=True,
        )
        csv_data = df[cols_order].to_csv(index=False)
        c2.download_button(
            "Download all (CSV)",
            data=csv_data,
            file_name="batch_extracted.csv",
            mime="text/csv",
            use_container_width=True,
        )

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
