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

# Default extraction schema configuration — invoice fields
DEFAULT_SCHEMA: list[dict] = [
    {"name": "invoice_no",      "field_type": "scalar", "enabled": True,
     "description": "Invoice number (labeled Invoice Number, Invoice #, Invoice No, No. — top right of invoice)"},
    {"name": "tracking_no",     "field_type": "scalar", "enabled": True,
     "description": "Tracking number (labeled Tracking Number)"},
    {"name": "pick_ticket_no",  "field_type": "scalar", "enabled": True,
     "description": "Pick ticket number (labeled Pick Ticket #)"},
    {"name": "po_no",           "field_type": "scalar", "enabled": True,
     "description": "Purchase order number (labeled Purchase Order Number, Customer PO, PO#, Cust PO, Customer PO No)"},
    {"name": "order_no",        "field_type": "scalar", "enabled": True,
     "description": "Order number (labeled Order Number, Sales Order, Sales Order #)"},
    {"name": "invoice_date",    "field_type": "scalar", "enabled": True,
     "description": "Invoice creation date (labeled Invoice Date, Invoicing Date, or Date). This is NOT the ship date or due date."},
    {"name": "term",            "field_type": "scalar", "enabled": True,
     "description": "Payment terms (labeled Terms, e.g., NET 45, NET 30, 2% NET 15)"},
    {"name": "pmt_method",      "field_type": "scalar", "enabled": True,
     "description": "Payment method (labeled Payment Method, e.g., Wire)"},
    {"name": "invoice_due",     "field_type": "scalar", "enabled": True,
     "description": "Invoice due date (labeled Invoice Due Date, Due)"},
    {"name": "sold_to",         "field_type": "scalar", "enabled": True,
     "description": "Billing address (labeled Sold To, Billing Address)"},
    {"name": "ship_to",         "field_type": "scalar", "enabled": True,
     "description": "Shipping address (labeled Ship To, Shipping Address)"},
    {"name": "weight",          "field_type": "scalar", "enabled": True,
     "description": "Weight (numeric value)"},
    {"name": "no_ctns",         "field_type": "scalar", "enabled": True,
     "description": "Number of containers (labeled Number of containers, # CTNS, Carto — numeric value)"},
    {"name": "order_date",      "field_type": "scalar", "enabled": True,
     "description": "Date the order was placed (labeled Order Date, Ordered, Order Placed). This is NOT the invoice date or ship date."},
    {"name": "ship_date",       "field_type": "scalar", "enabled": True,
     "description": "Date the goods were shipped (labeled Ship Date, Shipping Date, Ship, Shipped). This is NOT the invoice date or order date. Return null if not explicitly labeled as a ship date."},
    {"name": "ship_via",        "field_type": "scalar", "enabled": True,
     "description": "Shipping method (labeled Shipped Via, Shipping Method, Via, e.g., UPS, Truck, Pickup)"},
    {"name": "amount",          "field_type": "scalar", "enabled": True,
     "description": "Total amount due (labeled Total Amount, Total, Total Invoice, Merchandise Total, Total Due, Invoice Total)"},
    {"name": "sub_total",       "field_type": "scalar", "enabled": True,
     "description": "Sub total (numeric value)"},
    {"name": "freight",         "field_type": "scalar", "enabled": True,
     "description": "Freight charges (labeled Freight and Handling, Shipping — numeric value)"},
    {"name": "misc",            "field_type": "scalar", "enabled": True,
     "description": "Miscellaneous charges (labeled Additional charge, Addi Charge, Other Charge — numeric value)"},
    {"name": "sales_tax",       "field_type": "scalar", "enabled": True,
     "description": "Sales tax amount (numeric value)"},
    {"name": "discount",        "field_type": "scalar", "enabled": True,
     "description": "Discount amount (labeled Disc./Write-off — numeric value)"},
    {"name": "deposit",         "field_type": "scalar", "enabled": True,
     "description": "Paid/Deposit amount (labeled Paid, Payment — numeric value)"},
    {"name": "credit_applied",  "field_type": "scalar", "enabled": True,
     "description": "Credit applied (decimal value)"},
    {"name": "balance_due",     "field_type": "scalar", "enabled": True,
     "description": "Balance due (labeled Balance, Total balance Due — numeric value)"},
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

# Span extraction removed

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
    elements = []
    tables = _plumber_tables(path)
    
    # Extract line-by-line layout text using fitz
    pages_text = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            pages_text.append(page.get_text("text"))
    full_text = "\n\n".join(pages_text)
    
    return {"text": full_text, "markdown": None, "elements": elements,
            "tables": tables, "parser": "pymupdf"}

# OCR & Layout aware parsing mode
def parse_pdf_pymupdf4llm(path: Path) -> dict:
    chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    markdown = "\n\n".join(c["text"] for c in chunks)
    
    # Extract line-by-line layout text using fitz
    pages_text = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            pages_text.append(page.get_text("text"))
    raw_text = "\n\n".join(pages_text)
    
    # If the layout aware markdown parser missed/truncated content, fall back to/append layout text
    if len(markdown.strip()) < len(raw_text.strip()) * 0.7:
        text_source = f"{markdown}\n\n=== Raw Layout Text ===\n{raw_text}"
    else:
        text_source = markdown
        
    elements = []
    tables = _plumber_tables(path)
    return {"text": text_source, "markdown": markdown, "elements": elements,
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
    elements = []
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
        tables = _plumber_tables(tmp_path)
        tmp_path.unlink(missing_ok=True)
    except Exception:
        tables = []
    elements = []
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
        "You are extracting structured data from a document. "
        "Extract every field exactly as it appears in the document. Use null for missing values. "
        "For table/array fields extract ALL rows present. "
        "Preserve checkbox and selection markers exactly (e.g. ●, ○, ✓, ✗, X, checked, unchecked). "
        "For all numeric or monetary fields (amounts, totals, taxes, weights, quantities, etc.): "
        "return ONLY the plain numeric value — strip any currency symbols ($, €, £, ¥), "
        "commas used as thousands separators, and surrounding whitespace. "
        "Example: '$1,234.56' → '1234.56', '$ 500' → '500'."
    )
    if auto_tables:
        names = ", ".join(auto_tables)
        base += (
            f" For the table field(s) [{names}]: do NOT pre-assume column names. "
            "Detect all column headers from the actual table in the document, "
            "including any checkbox/mark columns, and return every row as a dict "
            "with those detected column names as keys."
        )
    
    # Add guidance for grid/table-structured scalar mapping
    base += (
        "\nNote: Many scalar fields (such as Terms, Due Date, Carton Count, Weight, and Invoice Totals) "
        "are often positioned in a table grid or side-by-side header-value pairs. "
        "Carefully map the values to their corresponding labels by reading the text layout. "
        "Ensure you do not map labels to values of adjacent unrelated fields."
    )
    return base

# Numeric fields that should be cleaned of currency symbols and commas
_NUMERIC_FIELD_HINTS = re.compile(
    r'(amount|total|subtotal|sub_total|tax|freight|misc|discount|deposit|credit|balance|weight|price|cost|charge|fee)',
    re.IGNORECASE,
)

def _clean_numeric_value(val: str) -> str:
    """Strip currency symbols and thousands-separator commas from a string that looks numeric."""
    if not isinstance(val, str):
        return val
    stripped = val.strip()
    # Remove currency symbols and leading/trailing whitespace
    cleaned = re.sub(r'^[\$€£¥\s]+', '', stripped).strip()
    # Remove commas used as thousands separators (only if number-like)
    if re.match(r'^-?[\d,]+(\.\d+)?$', cleaned):
        cleaned = cleaned.replace(',', '')
    return cleaned if cleaned else val

def _clean_numeric_fields(result: dict, schema: list[dict]) -> dict:
    """Post-process extracted result: clean currency symbols from numeric-looking scalar fields."""
    numeric_fields = {
        f["name"] for f in schema
        if f["field_type"] == "scalar" and f["enabled"]
        and (_NUMERIC_FIELD_HINTS.search(f["name"]) or _NUMERIC_FIELD_HINTS.search(f.get("description", "")))
    }
    cleaned = {}
    for k, v in result.items():
        if k in numeric_fields and isinstance(v, str):
            cleaned[k] = _clean_numeric_value(v)
        else:
            cleaned[k] = v
    return cleaned

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
    return _clean_numeric_fields(result.model_dump(), schema), time.time() - t0

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

    return _clean_numeric_fields(result, schema), time.time() - t0

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
