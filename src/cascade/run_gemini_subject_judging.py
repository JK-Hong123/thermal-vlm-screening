# -*- coding: utf-8 -*-
"""run_gemini_subject_judging.py — detached runner: wait until the free-tier daily quota resets (00:00 Pacific = 16:00 KST),
then (1) finish the Gemini subject's pass-2 (resumable), (2) judge its 80 reports with the Gemini judge (resumable),
(3) write results/gemini_subject_judging.DONE. Logs to results/gemini_subject_judging.log.
usage: GOOGLE_API_KEY=... python run_gemini_subject_judging.py [--at HH:MM]  (local time; default 16:03)
"""
import os, sys, time, subprocess, datetime, argparse
ap = argparse.ArgumentParser(); ap.add_argument('--at', default='16:03'); a = ap.parse_args()
R = r"<PROJECT_ROOT>\results"; S = r"<PROJECT_ROOT>\src\cascade"; PY = r"python"
LOG = open(os.path.join(R, 'gemini_subject_judging.log'), 'a', encoding='utf-8')
def log(m): LOG.write(f"{datetime.datetime.now():%H:%M:%S} {m}\n"); LOG.flush()
hh, mm = map(int, a.at.split(':')); now = datetime.datetime.now(); target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
if target < now: target += datetime.timedelta(days=1)
log(f"waiting until {target:%Y-%m-%d %H:%M}"); time.sleep(max(0, (target - now).total_seconds()))
env = dict(os.environ, PYTHONIOENCODING='utf-8')
for i in range(3):   # pass-2, resumable; repeat in case of residual per-minute 429s
    log(f"pass-2 attempt {i+1}"); p = subprocess.run([PY, 'gemini_pass2.py'], cwd=S, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
    log(p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-300:])
    n = len([f for f in os.listdir(os.path.join(R, 'vlm_out', 'v2gemini', 'v2')) if f.endswith('.txt')]); log(f"assembled reports: {n}")
    if n >= 80: break
    time.sleep(120)
for i in range(3):   # Gemini judge, forced-grounding protocol, all 80, resumable
    log(f"judge attempt {i+1}")
    p = subprocess.run([PY, 'judge_crossfamily.py', '--mode', 'run', '--provider', 'gemini', '--model', 'gemini-3.5-flash-lite', '--schema', 'full', '--tag', 'gemini_gemsubj',
                        '--key', os.path.join(R, 'judge_frontier', '_key_ext80.json'), '--v2_dir', os.path.join(R, 'vlm_out', 'v2gemini', 'v2')], cwd=S, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
    tail = [l for l in p.stdout.splitlines() if l.startswith('done:') or l.startswith('선호')]; log(' | '.join(tail) if tail else p.stderr[-300:])
    n = len([f for f in os.listdir(os.path.join(R, 'judge_crossfamily_gemini_gemsubj')) if f.startswith('case')]); log(f"judged: {n}")
    if n >= 80: break
    time.sleep(120)
open(os.path.join(R, 'gemini_subject_judging.DONE'), 'w').write(f"{datetime.datetime.now()}\n"); log("DONE")
