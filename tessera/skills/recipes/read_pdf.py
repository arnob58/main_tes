RECIPE = {
    "name": "read_pdf",
    "description": "Extract text from a PDF file",
    "keywords": ["read", "pdf", "extract", "resume", "document"],
}

SOURCE = '''import os

from pypdf import PdfReader

SKILL = {
    "name": "read_pdf",
    "description": "Extract the text of a PDF (first max_pages pages)",
    "parameters": {"path": "str - PDF path (supports ~)", "max_pages": "int - default 20"},
    "dependencies": ["pypdf"],
}


def run(path, max_pages=20, **_):
    path = os.path.abspath(os.path.expanduser(path))
    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages[:max_pages]]
    text = "\\n\\n".join(pages).strip()
    if not text:
        raise ValueError("no extractable text (the PDF may be scanned images)")
    return {"path": path, "pages": len(reader.pages), "text": text[:30000]}
'''


def build(request):
    return SOURCE
