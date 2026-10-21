RECIPE = {
    "name": "read_source",
    "description": "Read Python source files and summarise their structure with ast",
    "keywords": ["read", "source", "code", "python", "inspect", "own", "files"],
}

SOURCE = '''import ast
import os

SKILL = {
    "name": "read_source",
    "description": "Read a Python file or package: per-file classes, functions and imports via ast, plus file text when requested",
    "parameters": {
        "path": "str - .py file or folder",
        "include_text": "bool - include file contents (default False for folders, True for a single file)",
        "max_chars_per_file": "int - default 6000",
    },
}


def _outline(src):
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return {"syntax_error": f"line {exc.lineno}: {exc.msg}"}
    out = {"classes": [], "functions": [], "imports": []}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            out["classes"].append({"name": node.name, "methods": methods, "line": node.lineno})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            out["functions"].append({"name": node.name, "args": args, "line": node.lineno})
        elif isinstance(node, ast.Import):
            out["imports"].extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            out["imports"].append("." * node.level + (node.module or ""))
    return out


def run(path, include_text=None, max_chars_per_file=6000, **_):
    path = os.path.abspath(os.path.expanduser(path))
    files = []
    if os.path.isfile(path):
        files = [path]
        include_text = True if include_text is None else include_text
    else:
        for d, dirs, names in os.walk(path):
            dirs[:] = [x for x in dirs if not x.startswith((".", "__pycache__"))]
            files.extend(os.path.join(d, n) for n in sorted(names) if n.endswith(".py"))
    result = []
    for f in files[:200]:
        with open(f, encoding="utf-8", errors="replace") as fh:
            src = fh.read()
        entry = {"file": os.path.relpath(f, os.path.dirname(path)), "lines": src.count("\\n") + 1, **_outline(src)}
        if include_text:
            entry["text"] = src[:max_chars_per_file]
        result.append(entry)
    return {"root": path, "files": result}
'''


def build(request):
    return SOURCE
