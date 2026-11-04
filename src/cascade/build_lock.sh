#!/bin/sh
cd <PROJECT_ROOT>/src/cascade
PY=python
OUT='<PROJECT_ROOT>\results\cases_lock'
EX='<PROJECT_ROOT>\results\cases\index.jsonl'
rm -rf <PROJECT_ROOT>/results/cases_lock
$PY build_cases.py --seed 7 --per 14 --fps 8 --faccap 5 --out "$OUT" --exclude_index "$EX" && $PY session_agg.py --cases_dir "$OUT" && $PY render_v2.py --cases_dir "$OUT"
