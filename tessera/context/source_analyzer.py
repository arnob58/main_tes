"""Deterministic analysis of (possibly unfinished) Python source.

SRP: split source into the longest syntactically valid prefix that ends at a
top-level boundary, plus the exact remaining suffix — `prefix + suffix ==
source` always holds, so nothing is ever dropped — and extract the
interfaces (imports, classes, methods, function signatures) of the prefix.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FunctionSig:
    name: str
    signature: str
    doc: str = ""


@dataclass(frozen=True)
class ClassSig:
    name: str
    bases: str
    methods: tuple[FunctionSig, ...]
    doc: str = ""


@dataclass(frozen=True)
class SourceState:
    complete_prefix: str
    incomplete_suffix: str
    imports: tuple[str, ...] = ()
    classes: tuple[ClassSig, ...] = ()
    functions: tuple[FunctionSig, ...] = ()
    constants: tuple[str, ...] = ()
    parse_error: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not self.incomplete_suffix.strip()

    def interfaces_text(self) -> str:
        lines = list(self.imports)
        lines += [f"{c} = ..." for c in self.constants]
        for f in self.functions:
            lines.append(f"def {f.name}{f.signature}" + (f"  # {f.doc}" if f.doc else ""))
        for c in self.classes:
            lines.append(f"class {c.name}{c.bases}:" + (f"  # {c.doc}" if c.doc else ""))
            lines += [f"    def {m.name}{m.signature}" for m in c.methods]
        return "\n".join(lines)


def _first_doc_line(node) -> str:
    doc = ast.get_docstring(node) or ""
    return doc.strip().splitlines()[0][:80] if doc.strip() else ""


def _sig(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionSig:
    ret = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
    prefix = "async " if isinstance(fn, ast.AsyncFunctionDef) else ""
    return FunctionSig(prefix + fn.name, f"({ast.unparse(fn.args)}){ret}", _first_doc_line(fn))


def _top_level_boundaries(lines: list[str]) -> list[int]:
    """Line indices where a new top-level statement could start."""
    out = []
    for i, line in enumerate(lines):
        if line and not line[0].isspace() and not line.startswith(("#", ")", "]", "}")):
            out.append(i)
    return out


def split_complete(source: str, *, last_unit_suspect: bool = False) -> tuple[str, str, str]:
    """Return (prefix, suffix, parse_error). prefix + suffix == source.

    `last_unit_suspect`: generation was cut off, so even if everything parses
    the final top-level unit may be semantically unfinished; move it to the
    suffix unless the source ends cleanly with a newline after a blank line.
    """
    try:
        tree = ast.parse(source)
        error = ""
    except SyntaxError as exc:
        tree, error = None, f"line {exc.lineno}: {exc.msg}"

    lines = source.splitlines(keepends=True)
    if tree is not None and not last_unit_suspect:
        return source, "", ""
    if tree is not None and last_unit_suspect:
        if not tree.body:
            return source, "", ""
        last = tree.body[-1]
        start = last.lineno - 1
        if getattr(last, "decorator_list", None):
            start = min(d.lineno for d in last.decorator_list) - 1
        return "".join(lines[:start]), "".join(lines[start:]), ""

    for b in reversed(_top_level_boundaries(lines)):
        prefix = "".join(lines[:b])
        try:
            ast.parse(prefix)
        except SyntaxError:
            continue
        # don't split a decorator from its function
        while b > 0 and lines[b - 1].startswith("@"):
            b -= 1
        prefix = "".join(lines[:b])
        return prefix, "".join(lines[b:]), error
    return "", source, error


def analyze(source: str, *, last_unit_suspect: bool = False) -> SourceState:
    prefix, suffix, error = split_complete(source, last_unit_suspect=last_unit_suspect)
    assert prefix + suffix == source
    try:
        tree = ast.parse(prefix)
    except SyntaxError:  # pragma: no cover - prefix is validated above
        tree = ast.parse("")
    imports, classes, functions, consts = [], [], [], []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_sig(node))
        elif isinstance(node, ast.ClassDef):
            bases = f"({', '.join(ast.unparse(b) for b in node.bases)})" if node.bases else ""
            methods = tuple(_sig(n) for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
            classes.append(ClassSig(node.name, bases, methods, _first_doc_line(node)))
        elif isinstance(node, ast.Assign):
            consts.extend(t.id for t in node.targets if isinstance(t, ast.Name) and t.id.isupper())
    return SourceState(prefix, suffix, tuple(imports), tuple(classes), tuple(functions), tuple(consts), error)
