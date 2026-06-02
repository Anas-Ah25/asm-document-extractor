from __future__ import annotations
import re
import time
import base64
import json
import tempfile
import mimetypes
from pathlib import Path
from typing import Optional, List, Any

import fitz
import pdfplumber
import pymupdf4llm
import instructor
from openai import OpenAI
import pydantic
from pydantic import create_model, Field

# Graceful optional imports for DOCX parsing
try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None

try:
    from docx import Document as DocxDoc
except ImportError:
    DocxDoc = None

# Default extraction schema configuration
DEFAULT_SCHEMA: list[dict] = [
    {"name": "document_number", "field_type": "scalar", "enabled": True,
     "description": "Main document/drawing number e.g. 6518300, K9004-004400"},
    {"name": "revision", "field_type": "scalar", "enabled": True,
     "description": "Revision letter e.g. A, B"},
    {"name": "date", "field_type": "scalar", "enabled": True,
     "description": "Document date e.g. June 14, 2023"},
    {"name": "crm_number", "field_type": "scalar", "enabled": True,
     "description": "CRM number e.g. 23238, 24279, 29946"},
    {"name": "system_serial_number", "field_type": "scalar", "enabled": True,
     "description": "System serial number e.g. 6518300, K9004-004400"},
    {"name": "wbs_number", "field_type": "scalar", "enabled": True,
     "description": "WBS number e.g. 10FE.2876, 10EQM.03576"},
    {"name": "customer_name", "field_type": "scalar", "enabled": True,
     "description": "Customer company name e.g. Sony Semiconductor Corp."},
    {"name": "customer_location", "field_type": "scalar", "enabled": True,
     "description": "Customer location e.g. Atsugi Japan, Hillsboro Oregon"},
    {"name": "requested_ship_date", "field_type": "scalar", "enabled": True,
     "description": "Requested ship date e.g. October 25, 2023"},
    {"name": "sales_rep", "field_type": "scalar", "enabled": True,
     "description": "Sales representative name"},
    {"name": "ofe", "field_type": "scalar", "enabled": True,
     "description": "OFE field applications engineer name"},
    {"name": "bd_bmo_gpm", "field_type": "scalar", "enabled": True,
     "description": "BD / BMO / GPM contact name"},
    {
        "name": "process_modules",
        "field_type": "table",
        "enabled": True,
        "description": "All process module entries PM1 through PM6",
        "columns_mode": "defined",
        "columns": [
            {"name": "pm_position", "description": "Position e.g. PM1, PM2", "enabled": True},
            {"name": "module_type", "description": "Module type e.g. Synergis", "enabled": True},
            {"name": "serial_number", "description": "Module serial number", "enabled": True},
            {"name": "bu1_product_hierarchy", "description": "BU1 hierarchy e.g. Synergis MX IGZO", "enabled": True},
        ],
    },
]

# Dynamically construct Pydantic models from user configuration
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
                defs[f["name"]] = (
                    List[dict],
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
                        List[RowModel],
                        Field(default_factory=list, description=f["description"]),
                    )
    return create_model("ExtractionSchema", **defs)

# Extraction of span-level elements for citation anchoring
def _fitz_spans(path: Path) -> list[dict]:
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

# Geometry-based table parsing
def _plumber_tables(path: Path) -> list[dict]:
    tables: list[dict] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            for tbl in page.extract_tables():
                if tbl:
                    tables.append({"page": i, "rows": tbl})
    return tables

# Fast parsing mode
def parse_pdf_pymupdf(path: Path) -> dict:
    elements = _fitz_spans(path)
    tables = _plumber_tables(path)
    full_text = " ".join(e["text"] for e in elements)
    return {"text": full_text, "markdown": None, "elements": elements,
            "tables": tables, "parser": "pymupdf"}

# OCR & Layout aware parsing mode
def parse_pdf_pymupdf4llm(path: Path) -> dict:
    chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    markdown = "\n\n".join(c["text"] for c in chunks)
    elements = _fitz_spans(path)
    tables = _plumber_tables(path)
    return {"text": markdown, "markdown": markdown, "elements": elements,
            "tables": tables, "parser": "pymupdf4llm"}

# VLM page image parsing mode
def parse_pdf_gemini(path: Path, llm_base_url: str, api_key: str, model: str) -> dict:
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

# DOCX parsing fallback utility
def parse_docx(path: Path) -> dict:
    if DocumentConverter is not None:
        try:
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
            pass

    if DocxDoc is not None:
        doc2 = DocxDoc(str(path))
        elements, lines = [], []
        for para in doc2.paragraphs:
            t = para.text.strip()
            if t:
                elements.append({"text": t, "bbox": None, "page": None})
                lines.append(t)
        return {"text": "\n".join(lines), "markdown": None, "elements": elements,
                "tables": [], "parser": "python-docx"}

    raise ImportError("Neither docling nor python-docx is available for parsing DOCX documents.")

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}

# Raw image VLM extraction mode
def parse_image(path: Path, llm_base_url: str, api_key: str, model: str) -> dict:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    img_b64 = base64.b64encode(path.read_bytes()).decode()

    client = OpenAI(base_url=llm_base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Convert this engineering document image to clean markdown. "
                    "Preserve all tables as markdown tables with | separators. "
                    "Include all text exactly as shown. Do not add commentary."
                )},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
            ],
        }],
        temperature=0,
    )
    markdown = resp.choices[0].message.content or ""
    try:
        img_doc = fitz.open(str(path))
        pdfbytes = img_doc.convert_to_pdf()
        img_doc.close()
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(pdfbytes); tmp.close()
        tmp_path = Path(tmp.name)
        elements = _fitz_spans(tmp_path)
        tables = _plumber_tables(tmp_path)
        tmp_path.unlink(missing_ok=True)
    except Exception:
        elements, tables = [], []
    return {"text": markdown, "markdown": markdown, "elements": elements,
            "tables": tables, "parser": f"image-vision ({path.suffix})"}

# Main router for document parsing
def parse_document(path: Path, parser: str = "pymupdf4llm",
                   llm_base_url: str = "", api_key: str = "ollama", model: str = "") -> dict:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return parse_image(path, llm_base_url, api_key, model)
    if suffix != ".pdf":
        return parse_docx(path)
    if parser == "pymupdf":
        return parse_pdf_pymupdf(path)
    if parser == "gemini-vision":
        return parse_pdf_gemini(path, llm_base_url, api_key, model)
    return parse_pdf_pymupdf4llm(path)

# Fuzzy match value in document for spatial coordinates
def anchor(value: Optional[str], elements: list[dict]) -> Optional[dict]:
    if not value:
        return None
    v = str(value).lower().strip()
    for e in elements:
        if v in e["text"].lower():
            return {"page": e.get("page"), "bbox": e.get("bbox"),
                    "source_text": e["text"]}
    return None

# Attach spatial citations to extraction properties
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

_INFER_SYSTEM = """You are a document analyst. Given document text, suggest an extraction schema.
Return a JSON array of field objects. Each object must have:
  "name" (snake_case), "description" (what to extract), "field_type" ("scalar" or "table").
For table fields also include "columns": array of {name, description} objects.
Only suggest fields clearly present in this document type. Be concise."""

# LLM based schema inference helper
def infer_schema(parsed: dict, llm_base_url: str, api_key: str, model: str) -> list[dict]:
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
    result = []
    seen_fields = set()
    for f in suggested:
        raw_name = f.get("name", "field")
        name = raw_name.strip().lower().replace(" ", "_")
        if not name or name in seen_fields:
            continue
        seen_fields.add(name)
        entry: dict = {
            "name": name,
            "description": f.get("description", "") or raw_name,
            "field_type": f.get("field_type", "scalar"),
            "enabled": True,
        }
        if entry["field_type"] == "table":
            seen_cols = set()
            cols = []
            for c in f.get("columns", []):
                raw_cname = c.get("name", "col")
                cname = raw_cname.strip().lower().replace(" ", "_")
                if cname and cname not in seen_cols:
                    seen_cols.add(cname)
                    cols.append({
                        "name": cname,
                        "description": c.get("description", "") or raw_cname,
                        "enabled": True
                    })
            entry["columns"] = cols
        result.append(entry)
    return result

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

# Pydantic structured output extraction using Instructor
def extract_schema_ollama(
    parsed: dict, schema: list[dict], llm_base_url: str, api_key: str, model: str
) -> tuple[dict, float]:
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

# Base regex expressions for deterministic extraction
_REGEX: dict[str, str] = {
    "document_number": r'\b(\d{7}|K\d{4}-\d{6})\b',
    "revision": r'Rev(?:ision)?\.?\s+([A-Z])\b',
    "date": r'Date\s+([A-Za-z]+\s+\d{1,2},?\s*\d{4})',
    "crm_number": r'CRM\s*[#:\s-]*(\d{4,6})',
    "system_serial_number": r'System\s*(?:Serial\s*)?(?:Number|#)?\s*(\d{7}|K\d{4}-\d{6})',
    "wbs_number": r'\b(10[A-Z]+\.\d{4,7})\b',
    "requested_ship_date": r'Requested\s*Ship\s*Date\s+([A-Za-z]+\s+\d{1,2},?\s*\d{4})',
}
_LABEL_PAIRS: dict[str, list[str]] = {
    "customer_name": ["Customer Name"],
    "customer_location": ["Customer Location"],
    "sales_rep": ["Sales Rep", "Sales"],
    "ofe": ["OFE"],
    "bd_bmo_gpm": ["BD / BMO / GPM", "BD/BMO/GPM"],
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

# Regex-based layout heuristic extraction
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
                for tbl in parsed["tables"]:
                    tbl_rows = tbl["rows"]
                    if not tbl_rows:
                        continue
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

# Document metric exploration utility
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
