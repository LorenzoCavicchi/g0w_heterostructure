#!/usr/bin/env python3
"""Run the test suite without pytest:  python tests/run_tests.py [module ...]

Discovers ``test_*`` functions in ``tests/test_*.py`` (or the given modules) and
runs them in order, reporting pass/fail and timing.  With pytest installed,
``python -m pytest tests -q`` runs the same tests.
"""
import importlib.util
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

files = [HERE / (a if a.endswith(".py") else a + ".py") for a in sys.argv[1:]] or sorted(HERE.glob("test_*.py"))
n_ok = n_fail = 0
t_all = time.time()
for f in files:
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if getattr(mod, "SKIP_REASON", None):
        print(f"SKIP  {f.stem}: {mod.SKIP_REASON}", flush=True)
        continue
    for name in sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))):
        t0 = time.time()
        try:
            getattr(mod, name)()
            n_ok += 1
            print(f"PASS  {f.stem}::{name}  ({time.time()-t0:.1f} s)", flush=True)
        except Exception:
            n_fail += 1
            print(f"FAIL  {f.stem}::{name}  ({time.time()-t0:.1f} s)", flush=True)
            traceback.print_exc()
print(f"\n{n_ok} passed, {n_fail} failed in {time.time()-t_all:.1f} s")
sys.exit(1 if n_fail else 0)
