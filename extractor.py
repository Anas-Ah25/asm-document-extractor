"""
Extraction engine
Parsers  : pymupdf (fast) | pymupdf4llm (layout+OCR) | gemini-vision (VLM page images)
Backends : regex (A) | ollama (B)
Formats  : PDF | DOCX
"""
from __future__ import annotations
import re, time, base64, json, tempfile
from pathlib import Path
from typing import Optional, List, Any

import pydantic
from pydantic import create_model, Field


# ── Unified schema format ─────────────────────────────────────────────────────
#
#  Each entry is either a scalar field:
#    {"name": str, "description": str, "field_type": "scalar", "enabled": bool}
#  or a table field:
#    {"name": str, "description": str, "field_type": "table",  "enabled": bool,
#     "columns": [{"name": str, "description": str, "enabled": bool}, ...]}

DEFAULT_SCHEMA: list[dict] = [
    {"name": "document_number",      "field_type": "scalar", "enabled": True,
     "description": "Main document/drawing number e.g. 6518300, K9004-004400"},
    {"name": "revision",             "field_type": "scalar", "enabled": True,
     "description": "Revision letter e.g. A, B"},
    {"name": "date",                 "field_type": "scalar", "enabled": True,
     "description": "Document date e.g. June 14, 2023"},
    {"name": "crm_number",           "field_type": "scalar", "enabled": True,
     "description": "CRM number e.g. 23238, 24279, 29946"},
    {"name": "system_serial_number", "field_type": "scalar", "enabled": True,
     "description": "System serial number e.g. 6518300, K9004-004400"},
    {"name": "wbs_number",           "field_type": "scalar", "enabled": True,
     "description": "WBS number e.g. 10FE.2876, 10EQM.03576"},
    {"name": "customer_name",        "field_type": "scalar", "enabled": True,
     "description": "Customer company name e.g. Sony Semiconductor Corp."},
    {"name": "customer_location",    "field_type": "scalar", "enabled": True,
     "description": "Customer location e.g. Atsugi Japan, Hillsboro Oregon"},
    {"name": "requested_ship_date",  "field_type": "scalar", "enabled": True,
     "description": "Requested ship date e.g. October 25, 2023"},
    {"name": "sales_rep",            "field_type": "scalar", "enabled": True,
     "description": "Sales representative name"},
    {"name": "ofe",                  "field_type": "scalar", "enabled": True,
     "description": "OFE field applications engineer name"},
    {"name": "bd_bmo_gpm",           "field_type": "scalar", "enabled": True,
     "description": "BD / BMO / GPM contact name"},
    {
        "name": "process_modules",
        "field_type": "table",
        "enabled": True,
        "description": "All process module entries PM1 through PM6",
        # columns_mode="defined" → use the columns list below
        # columns_mode="auto"    → LLM infers columns; capture all columns including checkmarks
        "columns_mode": "defined",
        "columns": [
            {"name": "pm_position",           "description": "Position e.g. PM1, PM2",              "enabled": True},
            {"name": "module_type",           "description": "Module type e.g. Synergis",            "enabled": True},
            {"name": "serial_number",         "description": "Module serial number",                  "enabled": True},
            {"name": "bu1_product_hierarchy", "description": "BU1 hierarchy e.g. Synergis MX IGZO",  "enabled": True},
        ],
    },
]


# ── Dynamic Pydantic model builder ────────────────────────────────────────────

def build_pydantic_model(schema: list[dict]) -> type:
    defs: dict[str, Any] = {}
    for f in schema:
        if not f["enabled"]:
            continue
        if f["field_type"] == "scalar":
            defs[f["name"]] = (
                Optional[str],
                Field(default=None, description=f["description"]),
            )
        elif f["field_type"] == "table":
            mode = f.get("columns_mode", "defined")
            if mode == "auto":
                # LLM decides columns; each row is a free-form dict of strings.
                # Also captures checkbox/mark columns (e.g. "selected": "●").
                defs[f["name"]] = (
                    List[dict],  # type: ignore[valid-type]
                    Field(default_factory=list, description=f["description"]),
                )
            else:
                col_defs: dict[str, Any] = {}
                for col in f.get("columns", []):
                    if col["enabled"]:
                        col_defs[col["name"]] = (
                            Optional[str],
                            Field(default=None, description=col["description"]),
                        )
                if col_defs:
                    RowModel = create_model(f"{f['name']}_row", **col_defs)
                    defs[f["name"]] = (
                        List[RowModel],  # type: ignore[valid-type]
                        Field(default_factory=list, description=f["description"]),
                    )
    return create_model("ExtractionSchema", **defs)


# ── Parsers ───────────────────────────────────────────────────────────────────

def _fitz_spans(path: Path) -> list[dict]:
    """PyMuPDF span-level elements for citation anchoring — used by all PDF parsers."""
    import fitz
    elements: list[dict] = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        t = span.get("text", "").strip()
                        if t:
                            elements.append({
                                "text": t,
                                "bbox": list(span["bbox"]),
                                "page": page.number + 1,
                            })
    return elements


def _plumber_tables(path: Path) -> list[dict]:
    import pdfplumber
    tables: list[dict] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            for tbl in page.extract_tables():
                if tbl:
                    tables.append({"page": i, "rows": tbl})
    return tables


def parse_pdf_pymupdf(path: Path) -> dict:
    import fitz
    elements = _fitz_spans(path)
    tables = _plumber_tables(path)
    full_text = " ".join(e["text"] for e in elements)
    return {"text": full_text, "markdown": None, "elements": elements,
            "tables": tables, "parser": "pymupdf"}


def parse_pdf_pymupdf4llm(path: Path) -> dict:
    import pymupdf4llm
    chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    markdown = "\n\n".join(c["text"] for c in chunks)
    elements = _fitz_spans(path)
    tables = _plumber_tables(path)
    return {"text": markdown, "markdown": markdown, "elements": elements,
            "tables": tables, "parser": "pymupdf4llm"}


def parse_pdf_gemini(path: Path, llm_base_url: str, api_key: str, model: str) -> dict:
    """Render each page as an image → vision LLM → markdown."""
    import fitz
    from openai import OpenAI

    client = OpenAI(base_url=llm_base_url, api_key=api_key)
    pages_md: list[str] = []

    with fitz.open(str(path)) as doc:
        for page in doc:
            mat = fitz.Matrix(150 / 72, 150 / 72)
            pix = page.get_pixmap(matrix=mat)
            img_b64 = base64.b64encode(pix.tobytes("png")).decode()
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            "Convert this engineering document page to clean markdown. "
                            "Preserve all tables as markdown tables with | separators. "
                            "Include all text exactly as shown. Do not add commentary."
                        )},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                }],
                temperature=0,
            )
            pages_md.append(resp.choices[0].message.content or "")

    markdown = "\n\n---\n\n".join(pages_md)
    elements = _fitz_spans(path)
    tables = _plumber_tables(path)
    return {"text": markdown, "markdown": markdown, "elements": elements,
            "tables": tables, "parser": f"gemini-vision ({model})"}


def parse_docx(path: Path) -> dict:
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(str(path))
        doc = result.document
        md = doc.export_to_markdown()
        elements = []
        for item, _ in doc.iterate_items():
            t = getattr(item, "text", "") or ""
            if t.strip():
                elements.append({"text": t.strip(), "bbox": None, "page": None})
        return {"text": md, "markdown": md, "elements": elements,
                "tables": [], "parser": "docling"}
    except Exception:
        from docx import Document as DocxDoc
        doc2 = DocxDoc(str(path))
        elements, lines = [], []
        for para in doc2.paragraphs:
            t = para.text.strip()
            if t:
                elements.append({"text": t, "bbox": None, "page": None})
                lines.append(t)
        return {"text": "\n".join(lines), "markdown": None, "elements": elements,
                "tables": [], "parser": "python-docx"}


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}


def parse_image(path: Path) -> dict:
    """Wrap a raster image as a single fitz page and parse with pymupdf4llm (OCR)."""
    import fitz, pymupdf4llm
    # fitz can open images directly; embed as a PDF page for uniform handling
    img_doc = fitz.open(str(path))
    pdfbytes = img_doc.convert_to_pdf()
    img_doc.close()
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(pdfbytes); tmp.close()
    tmp_path = Path(tmp.name)
    try:
        chunks = pymupdf4llm.to_markdown(str(tmp_path), page_chunks=True)
        markdown = "\n\n".join(c["text"] for c in chunks)
        elements = _fitz_spans(tmp_path)
        tables   = _plumber_tables(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"text": markdown, "markdown": markdown, "elements": elements,
            "tables": tables, "parser": f"image-ocr ({path.suffix})"}


def parse_document(path: Path, parser: str = "pymupdf4llm",
                   llm_base_url: str = "", api_key: str = "ollama", model: str = "") -> dict:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return parse_image(path)
    if suffix != ".pdf":
        return parse_docx(path)
    if parser == "pymupdf":
        return parse_pdf_pymupdf(path)
    if parser == "gemini-vision":
        return parse_pdf_gemini(path, llm_base_url, api_key, model)
    return parse_pdf_pymupdf4llm(path)


# ── Citation anchoring ────────────────────────────────────────────────────────

def anchor(value: Optional[str], elements: list[dict]) -> Optional[dict]:
    if not value:
        return None
    v = str(value).lower().strip()
    for e in elements:
        if v in e["text"].lower():
            return {"page": e.get("page"), "bbox": e.get("bbox"),
                    "source_text": e["text"]}
    return None


def attach_citations(data: Any, elements: list[dict]) -> Any:
    if isinstance(data, dict):
        return {
            k: (
                {"value": v, "citation": anchor(v, elements)}
                if isinstance(v, str)
                else ([attach_citations(i, elements) for i in v] if isinstance(v, list)
                      else attach_citations(v, elements) if isinstance(v, dict)
                      else {"value": v, "citation": None})
            )
            for k, v in data.items()
        }
    return data


# ── Schema inference (Ollama) ─────────────────────────────────────────────────

_INFER_SYSTEM = """You are a document analyst. Given document text, suggest an extraction schema.
Return a JSON array of field objects. Each object must have:
  "name" (snake_case), "description" (what to extract), "field_type" ("scalar" or "table").
For table fields also include "columns": array of {name, description} objects.
Only suggest fields clearly present in this document type. Be concise."""


def infer_schema(parsed: dict, llm_base_url: str, api_key: str, model: str) -> list[dict]:
    from openai import OpenAI
    client = OpenAI(base_url=llm_base_url, api_key=api_key)
    text = (parsed.get("markdown") or parsed["text"])[:4000]
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _INFER_SYSTEM},
            {"role": "user", "content": f"Suggest extraction schema for:\n\n{text}"},
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content or "[]"
    raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    suggested = json.loads(raw)
    # Normalize
    result = []
    for f in suggested:
        entry: dict = {
            "name": f.get("name", "field"),
            "description": f.get("description", ""),
            "field_type": f.get("field_type", "scalar"),
            "enabled": True,
        }
        if entry["field_type"] == "table":
            entry["columns"] = [
                {"name": c.get("name", "col"), "description": c.get("description", ""), "enabled": True}
                for c in f.get("columns", [])
            ]
        result.append(entry)
    return result


# ── Backend B: Ollama + Instructor ────────────────────────────────────────────

def _build_extract_system(schema: list[dict]) -> str:
    auto_tables = [f["name"] for f in schema
                   if f["enabled"] and f["field_type"] == "table"
                   and f.get("columns_mode", "defined") == "auto"]
    base = (
        "You are extracting structured data from an engineering order specification. "
        "Extract every field exactly as it appears in the document. Use null for missing values. "
        "For table/array fields extract ALL rows present. "
        "Preserve checkbox and selection markers exactly (e.g. ●, ○, ✓, ✗, X, checked, unchecked)."
    )
    if auto_tables:
        names = ", ".join(auto_tables)
        base += (
            f" For the table field(s) [{names}]: do NOT pre-assume column names. "
            "Detect all column headers from the actual table in the document, "
            "including any checkbox/mark columns, and return every row as a dict "
            "with those detected column names as keys."
        )
    return base


def extract_schema_ollama(
    parsed: dict, schema: list[dict], llm_base_url: str, api_key: str, model: str
) -> tuple[dict, float]:
    import instructor
    from openai import OpenAI

    DynamicModel = build_pydantic_model(schema)
    client = instructor.from_openai(
        OpenAI(base_url=llm_base_url, api_key=api_key),
        mode=instructor.Mode.JSON,
    )
    text = parsed.get("markdown") or parsed["text"]
    table_ctx = "".join(
        f"\n[Table p{t['page']}]\n" + "\n".join(" | ".join(str(c or "") for c in r) for r in t["rows"])
        for t in parsed["tables"]
    )
    context = (text[:8000] + ("\n\nTABLES:\n" + table_ctx[:2000] if table_ctx else ""))[:10000]

    t0 = time.time()
    result = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _build_extract_system(schema)},
            {"role": "user", "content": f"Extract fields from:\n\n{context}"},
        ],
        response_model=DynamicModel,
        max_retries=2,
    )
    return result.model_dump(), time.time() - t0


# ── Backend A: Regex ──────────────────────────────────────────────────────────

_REGEX: dict[str, str] = {
    "document_number":      r'\b(\d{7}|K\d{4}-\d{6})\b',
    "revision":             r'Rev(?:ision)?\.?\s+([A-Z])\b',
    "date":                 r'Date\s+([A-Za-z]+\s+\d{1,2},?\s*\d{4})',
    "crm_number":           r'CRM\s*[#:\s-]*(\d{4,6})',
    "system_serial_number": r'System\s*(?:Serial\s*)?(?:Number|#)?\s*(\d{7}|K\d{4}-\d{6})',
    "wbs_number":           r'\b(10[A-Z]+\.\d{4,7})\b',
    "requested_ship_date":  r'Requested\s*Ship\s*Date\s+([A-Za-z]+\s+\d{1,2},?\s*\d{4})',
}
_LABEL_PAIRS: dict[str, list[str]] = {
    "customer_name":     ["Customer Name"],
    "customer_location": ["Customer Location"],
    "sales_rep":         ["Sales Rep", "Sales"],
    "ofe":               ["OFE"],
    "bd_bmo_gpm":        ["BD / BMO / GPM", "BD/BMO/GPM"],
}
_MOD_TYPES = ["Synergis", "Tession", "Valion", "Formion", "Prominis",
              "Side Blind", "Side blind", "Existing Module"]
_SKIP = re.compile(
    r'\b(Rev|Name|Location|Date|CRM|WBS|Sales|OFE|BD|BMO|GPM|Serial|System|Prepared|Module)\b', re.I
)


def _label_val(elements: list[dict], *labels: str) -> str:
    pat = re.compile("|".join(re.escape(l) for l in labels), re.IGNORECASE)
    for i, e in enumerate(elements):
        if pat.fullmatch(e["text"].strip()) or (pat.search(e["text"]) and len(e["text"]) < 40):
            for j in range(i + 1, min(i + 5, len(elements))):
                c = elements[j]["text"].strip()
                if c and not _SKIP.search(c) and len(c) < 80:
                    return c
    return ""


def extract_schema_regex(parsed: dict, schema: list[dict]) -> tuple[dict, float]:
    t0 = time.time()
    text, elements = parsed["text"], parsed["elements"]
    result: dict[str, Any] = {}

    for f in schema:
        if not f["enabled"]:
            continue
        name = f["name"]
        if f["field_type"] == "scalar":
            if name in _REGEX:
                m = re.search(_REGEX[name], text, re.IGNORECASE)
                result[name] = m.group(1).strip() if m else ""
            elif name in _LABEL_PAIRS:
                result[name] = _label_val(elements, *_LABEL_PAIRS[name])
            else:
                result[name] = ""
        elif f["field_type"] == "table":
            mode = f.get("columns_mode", "defined")
            rows: list[dict] = []

            if mode == "auto":
                # Return raw pdfplumber rows as dicts; first row = header if it looks like one
                for tbl in parsed["tables"]:
                    tbl_rows = tbl["rows"]
                    if not tbl_rows:
                        continue
                    # Use first row as header if it has no numeric-only cells
                    first = [str(c or "").strip() for c in tbl_rows[0]]
                    is_header = all(c and not re.fullmatch(r'[\d.]+', c) for c in first if c)
                    if is_header and len(tbl_rows) > 1:
                        headers = first
                        data_rows = tbl_rows[1:]
                    else:
                        headers = [f"col_{i}" for i in range(len(first))]
                        data_rows = tbl_rows
                    for row in data_rows:
                        cells = [str(c or "").strip() for c in row]
                        if any(cells):
                            rows.append(dict(zip(headers, cells)))
            else:
                cols_enabled = {c["name"] for c in f.get("columns", []) if c["enabled"]}
                seen: set[str] = set()
                pm_re = re.compile(r'\bPM\s*(\d)\b', re.IGNORECASE)
                for tbl in parsed["tables"]:
                    for row in tbl["rows"]:
                        row_text = " ".join(str(c or "") for c in row)
                        pm_m = pm_re.search(row_text)
                        if not pm_m:
                            continue
                        key = f"PM{pm_m.group(1)}"
                        if key in seen:
                            continue
                        seen.add(key)
                        serials = re.findall(r'\b(\d{7}|K\d{4}-\d{6})\b', row_text)
                        mod_type = next((m for m in _MOD_TYPES if m.lower() in row_text.lower()), "")
                        bu1 = next((str(c).strip() for c in row
                                    if c and re.search(r'Synergis\s+M[A-Z]', str(c), re.I)), None)
                        entry: dict[str, Any] = {}
                        if "pm_position" in cols_enabled:
                            entry["pm_position"] = key
                        if "module_type" in cols_enabled:
                            entry["module_type"] = mod_type
                        if "serial_number" in cols_enabled:
                            entry["serial_number"] = serials[0] if serials else None
                        if "bu1_product_hierarchy" in cols_enabled:
                            entry["bu1_product_hierarchy"] = bu1
                        rows.append(entry)

            result[name] = rows

    return result, time.time() - t0


# ── Full Explore ──────────────────────────────────────────────────────────────

def full_explore(parsed: dict) -> dict:
    text = parsed.get("markdown") or parsed["text"]
    return {
        "parser": parsed.get("parser"),
        "total_chars": len(text),
        "span_elements": len(parsed["elements"]),
        "tables_found": len(parsed["tables"]),
        "tables": [
            {"page": t["page"], "rows": len(t["rows"]),
             "cols": max((len(r) for r in t["rows"]), default=0),
             "data": t["rows"]}
            for t in parsed["tables"]
        ],
        "text": text,
    }
