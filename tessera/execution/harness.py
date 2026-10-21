"""Child-process harness that runs exactly one skill.

Invoked as:  python -I harness.py <skill.py> <args.json> <result.json>

Deliberately standalone (stdlib only, no Tessera imports) so it can run under
`python -I` — isolated mode ignores PYTHONPATH and the current directory,
so a stray file in the workspace can't shadow a stdlib module.

The result file is the only success signal the parent trusts:
  {"ok": true,  "result": <json>}
  {"ok": false, "error_type": "...", "message": "...", "traceback": "...", "missing_module": "..."}
"""

import importlib.util
import json
import sys
import traceback


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, (list, tuple, set)):
            return [_jsonable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        return repr(value)


def main() -> int:
    skill_path, args_path, result_path = sys.argv[1:4]
    # Isolated mode ignores PYTHONPATH; extra import roots must be explicit.
    import os
    for extra in filter(None, os.environ.get("TESSERA_SKILL_PATHS", "").split(os.pathsep)):
        sys.path.append(extra)
    with open(args_path, encoding="utf-8") as fh:
        args = json.load(fh)
    payload = None
    try:
        spec = importlib.util.spec_from_file_location("tessera_skill", skill_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        run = getattr(module, "run", None)
        if not callable(run):
            raise AttributeError("skill module has no callable run(**kwargs)")
        payload = {"ok": True, "result": _jsonable(run(**args))}
    except BaseException as exc:  # noqa: BLE001 — report everything, including SystemExit
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }
        if isinstance(exc, ModuleNotFoundError) and exc.name:
            payload["missing_module"] = exc.name
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
