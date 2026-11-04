# -*- coding: utf-8 -*-
"""run_gemini_full2_judging.py — re-run the cross-vendor (Gemini) judge under the CORRECTED forced-grounding protocol
(schema full2: stage 1 asks for the cyan marker's cell inside the white box) on all 80 locked cases for the three arms:
  frontier image-based  -> results/judge_crossfamily_gemini_full2/
  frontier no-image     -> results/judge_crossfamily_gemini_noimg_full2/
  Gemini subject        -> results/judge_crossfamily_gemini_gemsubj_full2/
Each arm is run twice (the second pass fills transient 429 failures; existing outputs are skipped). 240 requests in total
(free tier: 500/day per model). Logs to results/gemini_full2_judging.log; writes results/gemini_full2_judging.DONE.
usage: GOOGLE_API_KEY=... python run_gemini_full2_judging.py
"""
import os, subprocess, datetime
R = r"<PROJECT_ROOT>\results"; S = r"<PROJECT_ROOT>\src\cascade"; PY = r"python"
LOG = open(os.path.join(R, 'gemini_full2_judging.log'), 'a', encoding='utf-8')
def log(m): LOG.write(f"{datetime.datetime.now():%H:%M:%S} {m}\n"); LOG.flush()
env = dict(os.environ, PYTHONIOENCODING='utf-8')
ARMS = [('gemini_full2', None), ('gemini_noimg_full2', os.path.join(R, 'vlm_out', 'v2frontier', 'noimg')), ('gemini_gemsubj_full2', os.path.join(R, 'vlm_out', 'v2gemini', 'v2'))]
for tag, v2dir in ARMS:
    for attempt in (1, 2):
        cmd = [PY, 'judge_crossfamily.py', '--mode', 'run', '--provider', 'gemini', '--model', 'gemini-3.5-flash-lite', '--schema', 'full2', '--sleep', '2.5',
               '--tag', tag, '--key', os.path.join(R, 'judge_frontier', '_key_ext80.json')] + (['--v2_dir', v2dir] if v2dir else [])
        log(f"{tag} attempt {attempt}"); p = subprocess.run(cmd, cwd=S, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
        tail = [l for l in p.stdout.splitlines() if l.startswith('done:') or l.startswith('선호')]; log(' | '.join(tail) if tail else p.stderr[-300:])
        n = len([f for f in os.listdir(os.path.join(R, f'judge_crossfamily_{tag}')) if f.startswith('case')]); log(f"{tag} judged: {n}")
        if n >= 80: break
open(os.path.join(R, 'gemini_full2_judging.DONE'), 'w').write(f"{datetime.datetime.now()}\n"); log("DONE")
