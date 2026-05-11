"""
ASM Document Extractor
Parsers  : OCR (layout-aware) | Fast (no OCR)
Backends : A — Regex | B — Ollama LLM
"""
from __future__ import annotations
import json, copy, tempfile, time
from pathlib import Path

import streamlit as st

from extractor import (
    parse_document, full_explore,
    extract_schema_regex, extract_schema_ollama,
    infer_schema, attach_citations,
    DEFAULT_SCHEMA,
)

POOL_DIR = Path(__file__).parent
SUPPORTED = {".pdf", ".docx"}


# ── Page setup ────────────────────────────────────────────────────────────────

# Read from Streamlit secrets when deployed; fall back to local Ollama for dev.
_secrets = st.secrets if hasattr(st, "secrets") else {}
LLM_BASE_URL: str = _secrets.get("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY:      str = _secrets.get("API_KEY",      "ollama")
MODEL_NAME:   str = _secrets.get("MODEL_NAME",   "gemma4:31b-cloud")

st.set_page_config(page_title="ASM Extractor", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
.stTabs [data-baseweb="tab"] { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

if "schema" not in st.session_state:
    st.session_state.schema = copy.deepcopy(DEFAULT_SCHEMA)
if "results_cache" not in st.session_state:
    st.session_state.results_cache = {}
if "explore_result" not in st.session_state:
    st.session_state.explore_result = None
if "explore_parsed" not in st.session_state:
    st.session_state.explore_parsed = None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Settings")

    parser_choice = st.selectbox(
        "Parser",
        ["OCR — reads embedded images (recommended)", "Fast — no OCR"],
        help="OCR detects layout and reads text inside images. Slower but more accurate.",
    )
    parser_key = "pymupdf4llm" if parser_choice.startswith("OCR") else "pymupdf"

    backend_choice = st.radio(
        "Extraction backend",
        ["Ollama LLM (accurate)", "Regex (fast, no LLM)"],
    )
    use_llm = backend_choice.startswith("Ollama")

    st.divider()

    # ── Schema editor ──────────────────────────────────────────────────────────
    st.subheader("Target Schema")

    schema = st.session_state.schema
    scalar_fields = [f for f in schema if f["field_type"] == "scalar"]
    table_fields  = [f for f in schema if f["field_type"] == "table"]

    with st.expander("Scalar fields", expanded=True):
        for f in scalar_fields:
            idx = schema.index(f)
            col_cb, col_name = st.columns([1, 5])
            schema[idx]["enabled"] = col_cb.checkbox(
                f["name"], value=f["enabled"], key=f"sf_{f['name']}",
                label_visibility="collapsed",
            )
            col_name.caption(f"**{f['name']}** — {f['description'][:50]}")

        with st.form("add_scalar", clear_on_submit=True):
            sname = st.text_input("Field name", placeholder="e.g. po_number")
            sdesc = st.text_input("Description", placeholder="Purchase order number")
            if st.form_submit_button("Add scalar field") and sname.strip():
                schema.append({
                    "name": sname.strip().lower().replace(" ", "_"),
                    "description": sdesc.strip() or sname.strip(),
                    "field_type": "scalar",
                    "enabled": True,
                })
                st.rerun()

    with st.expander("Table fields", expanded=True):
        for f in table_fields:
            idx = schema.index(f)
            col_cb, col_name = st.columns([1, 5])
            schema[idx]["enabled"] = col_cb.checkbox(
                f["name"], value=f["enabled"], key=f"tf_{f['name']}",
                label_visibility="collapsed",
            )
            col_name.caption(f"**{f['name']}** — {f['description'][:40]}")

            if f["enabled"]:
                cur_mode = f.get("columns_mode", "defined")
                mode_label = st.radio(
                    "Columns",
                    ["Defined", "Auto-detect (LLM infers columns + marks)"],
                    index=0 if cur_mode == "defined" else 1,
                    key=f"cmode_{f['name']}",
                    horizontal=True,
                )
                schema[idx]["columns_mode"] = "defined" if mode_label == "Defined" else "auto"

                if schema[idx]["columns_mode"] == "defined":
                    with st.expander(f"Columns of {f['name']}", expanded=False):
                        for ci, col in enumerate(f.get("columns", [])):
                            schema[idx]["columns"][ci]["enabled"] = st.checkbox(
                                f"{col['name']} — {col['description'][:40]}",
                                value=col["enabled"],
                                key=f"col_{f['name']}_{col['name']}",
                            )
                else:
                    st.caption("LLM will detect all columns including checkbox/mark columns.")

        with st.form("add_table", clear_on_submit=True):
            tname = st.text_input("Table field name", placeholder="e.g. line_items")
            tdesc = st.text_input("Description", placeholder="Order line items")
            tauto = st.checkbox(
                "Auto-detect columns (LLM infers columns + marks)",
                value=False,
                help="Recommended when column layout is unknown or includes checkboxes.",
            )
            tcols = st.text_area(
                "Columns (one per line: name — description) — ignored if auto-detect",
                placeholder="item_number — Item number\nquantity — Quantity ordered",
            )
            if st.form_submit_button("Add table field") and tname.strip():
                cols = []
                if not tauto:
                    for line in tcols.strip().splitlines():
                        parts = line.split("—", 1)
                        cols.append({
                            "name": parts[0].strip().lower().replace(" ", "_"),
                            "description": parts[1].strip() if len(parts) > 1 else parts[0].strip(),
                            "enabled": True,
                        })
                schema.append({
                    "name": tname.strip().lower().replace(" ", "_"),
                    "description": tdesc.strip() or tname.strip(),
                    "field_type": "table",
                    "enabled": True,
                    "columns_mode": "auto" if tauto else "defined",
                    "columns": cols,
                })
                st.rerun()

    col_reset, col_infer = st.columns(2)
    if col_reset.button("Reset schema", use_container_width=True):
        st.session_state.schema = copy.deepcopy(DEFAULT_SCHEMA)
        st.rerun()

    if col_infer.button("Infer schema", use_container_width=True,
                        help="Parse a document and let the LLM suggest a schema"):
        st.session_state._infer_pending = True

    if st.session_state.get("_infer_pending"):
        pool_files = sorted(
            [f for f in POOL_DIR.iterdir() if f.suffix.lower() in SUPPORTED
             and f.stem not in ("app", "extractor")],
            key=lambda f: f.name,
        )
        if pool_files:
            pick = st.selectbox("Pick sample doc for inference", [f.name for f in pool_files],
                                key="infer_pick")
            if st.button("Run inference", key="run_infer"):
                with st.spinner("Parsing + inferring schema…"):
                    parsed = parse_document(
                        POOL_DIR / pick, parser=parser_key,
                        llm_base_url=LLM_BASE_URL, api_key=API_KEY, model=MODEL_NAME,
                    )
                    suggested = infer_schema(parsed, LLM_BASE_URL, API_KEY, MODEL_NAME)
                if suggested:
                    st.session_state.schema = suggested
                    st.session_state._infer_pending = False
                    st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_pool_files() -> list[Path]:
    return sorted(
        [f for f in POOL_DIR.iterdir()
         if f.suffix.lower() in SUPPORTED and f.stem not in ("app", "extractor")],
        key=lambda f: f.name,
    )


def run_extraction(path: Path) -> tuple[dict, float]:
    parsed = parse_document(path, parser=parser_key,
                            llm_base_url=LLM_BASE_URL, api_key=API_KEY, model=MODEL_NAME)
    if use_llm:
        result, elapsed = extract_schema_ollama(
            parsed, st.session_state.schema, LLM_BASE_URL, API_KEY, MODEL_NAME,
        )
    else:
        result, elapsed = extract_schema_regex(parsed, st.session_state.schema)
    return attach_citations(result, parsed["elements"]), elapsed


def show_result(result: dict, elapsed: float, fname: str):
    backend_label = f"Ollama ({MODEL_NAME})" if use_llm else "Regex"
    st.success(f"{fname} — {parser_key} + {backend_label} — {elapsed:.1f}s")

    schema = st.session_state.schema
    scalar_names = {f["name"] for f in schema if f["field_type"] == "scalar" and f["enabled"]}
    table_names  = {f["name"] for f in schema if f["field_type"] == "table"  and f["enabled"]}

    # Scalar cards
    scalar_items = {k: v for k, v in result.items() if k in scalar_names}
    if scalar_items:
        cols = st.columns(3)
        for ci, (field, wrapped) in enumerate(scalar_items.items()):
            val  = wrapped.get("value", "") if isinstance(wrapped, dict) else wrapped
            cite = wrapped.get("citation") if isinstance(wrapped, dict) else None
            with cols[ci % 3]:
                st.markdown(f"**{field}**")
                st.code(val or "—", language=None)
                if cite:
                    pg   = cite.get("page")
                    src  = cite.get("source_text", "")[:40]
                    st.caption(f"p.{pg} · {src!r}" if pg else src)

    # Table fields
    for field in table_names:
        if field not in result:
            continue
        rows_raw = result[field]
        if not rows_raw:
            st.info(f"No rows found for **{field}**")
            continue
        import pandas as pd
        rows = [
            {k: (v.get("value", "") if isinstance(v, dict) else v)
             for k, v in row.items()}
            for row in rows_raw
        ]
        st.markdown(f"**{field}**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Download + raw JSON
    col_dl, col_raw = st.columns([1, 4])
    with col_dl:
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2, default=str),
            file_name=f"{Path(fname).stem}_extracted.json",
            mime="application/json",
            key=f"dl_{fname}_{time.time()}",
        )
    with col_raw:
        with st.expander("Raw JSON"):
            st.json(result, expanded=1)


def flatten_scalar(result: dict) -> dict:
    row = {}
    schema = st.session_state.schema
    scalar_names = {f["name"] for f in schema if f["field_type"] == "scalar" and f["enabled"]}
    for k, v in result.items():
        if k not in scalar_names:
            continue
        row[k] = v.get("value", "") if isinstance(v, dict) else v
    return row


# ── Main tabs ─────────────────────────────────────────────────────────────────

st.title("ASM Document Extractor")
tab_schema, tab_explore = st.tabs(["Schema Extract", "Full Explore"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Schema Extract
# ═══════════════════════════════════════════════════════════════════════════════

with tab_schema:
    mode_col, _ = st.columns([2, 3])
    with mode_col:
        source_mode = st.radio("Source", ["Pool (batch)", "Single document"], horizontal=True)
    st.divider()

    # ── Pool mode ──────────────────────────────────────────────────────────────
    if source_mode.startswith("Pool"):
        pool_files = get_pool_files()

        if not pool_files:
            st.info("No PDF/DOCX files found in the pool folder.")
        else:
            st.markdown(f"**{len(pool_files)} document(s) in pool**")
            selected: list[Path] = []
            c0, c1, c2 = st.columns([5, 1, 1])
            c0.markdown("**File**")
            c1.markdown("**Size**")
            c2.markdown("**Select**")
            for fpath in pool_files:
                ca, cb, cc = st.columns([5, 1, 1])
                status = "[done]" if fpath.name in st.session_state.results_cache else ""
                ca.write(f"{status} {fpath.name}")
                cb.caption(f"{fpath.stat().st_size // 1024} KB")
                if cc.checkbox("sel", key=f"sel_{fpath.name}", label_visibility="collapsed"):
                    selected.append(fpath)

            btn1, btn2, btn3 = st.columns([2, 2, 4])
            run_selected = btn1.button("Extract selected", disabled=not selected)
            run_all      = btn2.button("Extract all")
            if btn3.button("Clear results"):
                st.session_state.results_cache = {}
                st.rerun()

            targets = selected if run_selected else (pool_files if run_all else [])
            if targets:
                prog = st.progress(0, text="Starting…")
                for idx, fpath in enumerate(targets):
                    prog.progress(idx / len(targets), text=f"Processing {fpath.name}…")
                    try:
                        with st.spinner(f"Parsing {fpath.name}…"):
                            result, elapsed = run_extraction(fpath)
                        st.session_state.results_cache[fpath.name] = {
                            "result": result, "elapsed": elapsed,
                        }
                    except Exception as exc:
                        st.session_state.results_cache[fpath.name] = {"error": str(exc)}
                prog.progress(1.0, text="Done")

            if st.session_state.results_cache:
                st.divider()
                st.subheader("Results")

                import pandas as pd
                table_rows = []
                for fname, data in st.session_state.results_cache.items():
                    if "error" in data:
                        table_rows.append({"file": fname, "error": data["error"]})
                    else:
                        row = flatten_scalar(data["result"])
                        row["file"] = fname
                        row["time_s"] = f"{data['elapsed']:.1f}s"
                        table_rows.append(row)

                if table_rows:
                    df = pd.DataFrame(table_rows)
                    cols_order = ["file"] + [c for c in df.columns if c != "file"]
                    st.dataframe(df[cols_order], use_container_width=True, hide_index=True)

                all_results = {
                    fn: d.get("result", {"error": d.get("error")})
                    for fn, d in st.session_state.results_cache.items()
                }
                st.download_button(
                    "Download all (JSON)",
                    data=json.dumps(all_results, indent=2, default=str),
                    file_name="batch_extracted.json",
                    mime="application/json",
                )

                with st.expander("Per-document detail"):
                    for fname, data in st.session_state.results_cache.items():
                        st.markdown(f"### {fname}")
                        if "error" in data:
                            st.error(data["error"])
                        else:
                            show_result(data["result"], data["elapsed"], fname)

    # ── Single mode ────────────────────────────────────────────────────────────
    else:
        left, right = st.columns([1, 2])
        with left:
            src = st.radio("Pick document", ["From pool", "Upload"])
            chosen_path: Path | None = None
            chosen_name = ""

            if src == "From pool":
                pool_files2 = get_pool_files()
                chosen_name = st.selectbox("Select file", [f.name for f in pool_files2])
                chosen_path = (POOL_DIR / chosen_name) if chosen_name else None
            else:
                uploaded = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
                if uploaded:
                    suffix = Path(uploaded.name).suffix.lower()
                    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                    tmp.write(uploaded.read())
                    tmp.close()
                    chosen_path = Path(tmp.name)
                    chosen_name = uploaded.name

            extract_btn = st.button("Extract", type="primary", disabled=not chosen_path)

        with right:
            if extract_btn and chosen_path:
                try:
                    with st.spinner(f"Extracting {chosen_name}…"):
                        result, elapsed = run_extraction(chosen_path)
                    st.session_state.single_result = (result, elapsed, chosen_name)
                except Exception as exc:
                    st.error(f"Error: {exc}")

            if "single_result" in st.session_state:
                res, elapsed, fname = st.session_state.single_result
                show_result(res, elapsed, fname)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Full Explore
# ═══════════════════════════════════════════════════════════════════════════════

with tab_explore:
    st.markdown("Parse a document and view all its structured content — text, tables, layout.")
    left2, right2 = st.columns([1, 2])

    with left2:
        exp_src = st.radio("Pick document", ["From pool", "Upload"], key="exp_src")
        exp_path: Path | None = None
        exp_name = ""

        if exp_src == "From pool":
            pool_exp = get_pool_files()
            exp_name = st.selectbox("Select file", [f.name for f in pool_exp], key="exp_sel")
            exp_path = (POOL_DIR / exp_name) if exp_name else None
        else:
            exp_up = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"], key="exp_up")
            if exp_up:
                sfx = Path(exp_up.name).suffix.lower()
                tmp2 = tempfile.NamedTemporaryFile(suffix=sfx, delete=False)
                tmp2.write(exp_up.read())
                tmp2.close()
                exp_path = Path(tmp2.name)
                exp_name = exp_up.name

        explore_btn = st.button("Parse & Explore", type="primary", disabled=not exp_path)

    with right2:
        if explore_btn and exp_path:
            with st.spinner(f"Parsing with {parser_key}…"):
                ep = parse_document(exp_path, parser=parser_key,
                                    llm_base_url=LLM_BASE_URL, api_key=API_KEY, model=MODEL_NAME)
            st.session_state.explore_result = full_explore(ep)
            st.session_state.explore_parsed = ep

        if st.session_state.explore_result:
            er = st.session_state.explore_result

            m1, m2, m3 = st.columns(3)
            m1.metric("Characters",    f"{er['total_chars']:,}")
            m2.metric("Span elements", er["span_elements"])
            m3.metric("Tables found",  er["tables_found"])

            out_tabs = st.tabs(["Full text", "Tables", "Raw JSON"])

            with out_tabs[0]:
                preview = er["text"][:15000]
                if len(er["text"]) > 15000:
                    preview += "\n\n…(truncated)"
                st.markdown(preview)
                st.download_button(
                    "Download full text",
                    data=er["text"],
                    file_name="full_text.md",
                    mime="text/markdown",
                )

            with out_tabs[1]:
                if not er["tables"]:
                    st.info("No tables detected.")
                for tbl in er["tables"]:
                    st.markdown(f"**Table — page {tbl['page']} ({tbl['rows']} rows × {tbl['cols']} cols)**")
                    rows = tbl.get("data", [])
                    if rows:
                        import pandas as pd
                        header = rows[0]
                        data_rows = rows[1:]
                        try:
                            df = pd.DataFrame(data_rows, columns=header)
                        except Exception:
                            df = pd.DataFrame(rows)
                        st.dataframe(df, use_container_width=True, hide_index=True)

            with out_tabs[2]:
                st.download_button(
                    "Download full JSON",
                    data=json.dumps(er, indent=2, default=str),
                    file_name="full_explore.json",
                    mime="application/json",
                )
                st.json(er, expanded=1)
