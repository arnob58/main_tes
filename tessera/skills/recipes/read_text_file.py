RECIPE = {
    "name": "read_text_file",
    "description": "Read a plain-text file (txt, md, csv, json, py, …)",
    "keywords": ["read", "open", "text", "file", "txt", "markdown", "csv", "json"],
}

SOURCE = '''import os

SKILL = {
    "name": "read_text_file",
    "description": "Read a UTF-8 text file and return its contents (truncated to max_chars)",
    "parameters": {"path": "str - file path (supports ~)", "max_chars": "int - default 20000"},
}


def run(path, max_chars=20000, **_):
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read(max_chars + 1)
    return {"path": path, "text": text[:max_chars], "truncated": len(text) > max_chars}
'''


def build(request):
    return SOURCE
