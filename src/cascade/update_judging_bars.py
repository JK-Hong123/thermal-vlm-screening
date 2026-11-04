# -*- coding: utf-8 -*-
"""update_judging_bars.py — append (or refresh) the Gemini-SUBJECT bars in results/judging_bars.json (read by Fig. 'judging'):
the low-cost commercial model's reports (results/vlm_out/v2gemini/v2) judged against the code-only report by
  (i) the Gemini judge, forced-grounding protocol, 80 cases   -> results/judge_crossfamily_gemini_gemsubj/
  (ii) the Claude judge, same protocol, 32 cases               -> results/judge_crossfamily_gemsubj_claude/
Preference = judge's preferred letter mapped to the arm through the blinding key; CI = bootstrap over cases (2000 resamples).
"""
import json, os, sys, io, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np
R = r"<PROJECT_ROOT>\results"; BARS = os.path.join(R, 'judging_bars.json')
GROUP = 'Low-cost commercial subject (Gemini), judges without the facts'
def tally(dirn, keyf):
    key = json.load(open(keyf, encoding='utf-8')); wins = []
    for cid in sorted(key):
        fp = os.path.join(dirn, cid + '.json')
        if not os.path.exists(fp): continue
        j = json.load(open(fp, encoding='utf-8')); p = j.get('preferred_report')
        arm = 'tie' if p == 'tie' else (key[cid]['A'] if p == 'A' else key[cid]['B'])
        wins.append(arm == 'v2')
    if not wins: return None
    rng = np.random.default_rng(0); w = np.array(wins, dtype=float); n = len(w)
    bs = [w[rng.integers(0, n, n)].mean() for _ in range(2000)]
    return dict(k=int(w.sum()), n=n, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
items = [('Gemini subject, Claude judge (n=%d)', os.path.join(R, 'judge_crossfamily_gemsubj_claude'), os.path.join(R, 'judge_frontier', '_key.json'), '#2ca02c'),
         ('Gemini subject, Gemini judge (n=%d)', os.path.join(R, 'judge_crossfamily_gemini_gemsubj'), os.path.join(R, 'judge_frontier', '_key_ext80.json'), '#98df8a')]
J = [b for b in json.load(open(BARS, encoding='utf-8')) if b['group'] != GROUP]
for label, d, keyf, col in items:
    t = tally(d, keyf)
    if t is None: print('no results yet:', d); continue
    J.append(dict(group=GROUP, label=label % t['n'], k=t['k'], n=t['n'], lo=round(t['lo'], 3), hi=round(t['hi'], 3), color=col)); print(label % t['n'], f"{t['k']}/{t['n']} [{t['lo']:.3f},{t['hi']:.3f}]")
json.dump(J, open(BARS, 'w', encoding='utf-8'), ensure_ascii=False, indent=1); print('bars:', len(J))
