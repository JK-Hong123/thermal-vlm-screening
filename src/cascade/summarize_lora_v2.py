# -*- coding: utf-8 -*-
"""summarize_lora_v2.py — one table over every results/lora_v2/<tag>/history.json (base -> last epoch, val and locked)."""
import json, os, sys, io, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = r"<PROJECT_ROOT>\results\lora_v2"
rows = []
for hp in sorted(glob.glob(os.path.join(ROOT, '*', 'history.json'))):
    h = json.load(open(hp, encoding='utf-8')); a = h['args']; tag = os.path.basename(os.path.dirname(hp))
    b = h['base']; last = h['epochs'][-1]['eval'] if h['epochs'] else None
    def f(e, k): return e.get(k, float('nan')) if e else float('nan')
    rows.append(dict(tag=tag, render=a['render'], label=a['label'], targets=a['targets'], task=a['task'], n=a['n'],
                     v_acc0=f(b['val'], 'acc'), v_k0=f(b['val'], 'kappa'), v_acc=f(last['val'], 'acc') if last else float('nan'), v_k=f(last['val'], 'kappa') if last else float('nan'),
                     l_acc0=f(b['lock'], 'acc'), l_k0=f(b['lock'], 'kappa'), l_acc=f(last['lock'], 'acc') if last else float('nan'), l_k=f(last['lock'], 'kappa') if last else float('nan'),
                     u_k=f(last['lock'], 'unseen_kappa') if last else float('nan'), dist=f(last['val'], 'mean_dist') if last else float('nan')))
print(f"{'tag':24s} {'render':7s} {'label':6s} {'targets':11s} {'task':5s} {'n':>5s} | {'val acc0→acc':>14s} {'val κ0→κ':>12s} | {'lock acc0→acc':>15s} {'lock κ0→κ':>12s} {'unseen κ':>8s} {'dist':>5s}")
for r in rows:
    print(f"{r['tag']:24s} {r['render']:7s} {r['label']:6s} {r['targets']:11s} {r['task']:5s} {r['n']:5d} | {r['v_acc0']:.3f}→{r['v_acc']:.3f}   {r['v_k0']:.3f}→{r['v_k']:.3f} | {r['l_acc0']:.3f}→{r['l_acc']:.3f}    {r['l_k0']:.3f}→{r['l_k']:.3f}   {r['u_k']:.3f}  {r['dist']:.3f}")
print("\nchance: acc 0.111, κ 0 | previous recipe (fixed render, argmax label, marked+unmarked mixed): unmarked val κ 0.275, locked κ 0.29")
