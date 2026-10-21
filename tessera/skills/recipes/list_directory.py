RECIPE = {
    "name": "list_directory",
    "description": "List the entries of a folder",
    "keywords": ["list", "directory", "folder", "ls", "contents"],
}

SOURCE = '''import os

SKILL = {
    "name": "list_directory",
    "description": "List files and folders in a directory with sizes",
    "parameters": {"path": "str - folder (supports ~), default .", "show_hidden": "bool - default False"},
}


def run(path=".", show_hidden=False, **_):
    path = os.path.abspath(os.path.expanduser(path))
    entries = []
    for e in sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower())):
        if not show_hidden and e.name.startswith("."):
            continue
        entries.append({"name": e.name, "type": "dir" if e.is_dir() else "file",
                        "size": e.stat().st_size if e.is_file() else None})
    return {"path": path, "entries": entries[:500]}
'''


def build(request):
    return SOURCE
