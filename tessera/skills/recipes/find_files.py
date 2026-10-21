RECIPE = {
    "name": "find_files",
    "description": "Search folders for files by name, glob pattern or extension",
    "keywords": ["find", "search", "locate", "files", "file"],
}

SOURCE = '''import fnmatch
import os
import time

SKILL = {
    "name": "find_files",
    "description": "Search a folder tree for files by name fragment, glob pattern or extension",
    "parameters": {
        "root": "str - folder to search, default ~ (supports ~)",
        "pattern": "str - glob such as *.pdf, default *",
        "name_contains": "str - case-insensitive fragment of the file name",
        "extensions": "list[str] - e.g. ['.pdf', '.docx']",
        "max_results": "int - default 50",
        "max_depth": "int - default 6",
    },
}

SKIP = {".git", "node_modules", "__pycache__", ".venv", "venv", "AppData", "Library", ".cache", "site-packages"}


def run(root="~", pattern="*", name_contains="", extensions=None, max_results=50, max_depth=6, **_):
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"folder not found: {root}")
    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in (extensions or [])}
    frag = name_contains.lower()
    base_depth = root.rstrip(os.sep).count(os.sep)
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.count(os.sep) - base_depth
        dirnames[:] = [d for d in dirnames if d not in SKIP and not d.startswith(".")] if depth < max_depth else []
        for fn in filenames:
            low = fn.lower()
            if not fnmatch.fnmatch(low, pattern.lower()):
                continue
            if frag and frag not in low:
                continue
            if exts and os.path.splitext(low)[1] not in exts:
                continue
            path = os.path.join(dirpath, fn)
            try:
                st = os.stat(path)
            except OSError:
                continue
            found.append({"path": path, "size": st.st_size,
                          "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))})
            if len(found) >= max_results:
                return {"root": root, "matches": found, "truncated": True}
    return {"root": root, "matches": found, "truncated": False}
'''


def build(request):
    return SOURCE
