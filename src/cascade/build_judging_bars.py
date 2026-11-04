# -*- coding: utf-8 -*-
"""build_judging_bars.py — rebuild results/judging_bars.json (read by the judging figure) from the per-case judge outputs.
Groups = judging protocol; bars = subject arms. Preference = judge's preferred letter mapped to the arm through the
blinding key; CI = bootstrap over cases (2000 resamples, seed 0). The 3B and adapter rows have no per-case letter files in
this format and are entered from their summaries (0/34, 0/32).
"""
import json, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np
R = r"<PROJECT_ROOT>\results"; K32 = os.path.join(R, 'judge_frontier', '_key.json'); K80 = os.path.join(R, 'judge_frontier', '_key_ext80.json')
def tally(dirn, keyf):
    key = json.load(open(keyf, encoding='utf-8')); wins = []
    for cid in sorted(key):
        fp = os.path.join(R, dirn, cid + '.json')
        if not os.path.exists(fp): continue
        j = json.load(open(fp, encoding='utf-8')); p = j.get('preferred_report')
        wins.append((key[cid][p] if p in ('A', 'B') else 'tie') == 'v2')
    rng = np.random.default_rng(0); w = np.array(wins, dtype=float); n = len(w); k = int(w.sum())
    if k in (0, n):   # a bootstrap interval collapses to a point for a degenerate proportion; use the exact (Clopper-Pearson) interval
        from scipy.stats import beta
        lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)); hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)); return k, n, lo, hi
    bs = [w[rng.integers(0, n, n)].mean() for _ in range(2000)]
    return k, n, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
PUR, PUR_L, GRN, GRN_L, RED, ORA = '#9467bd', '#c5b0d5', '#2ca02c', '#98df8a', '#d62728', '#ff7f0e'
SPEC = [
 ('Same-model judge told which report is model-written, facts given (September protocol)', [
   ('Qwen2.5-VL-3B, zero-shot (n=34)', None, (0, 34, 0.0, 0.103), RED),
   ('3B + marker-reading adapter (n=32)', None, (0, 32, 0.0, 0.109), ORA),
   ('Frontier, September reports (n=32)', ('judge_frontier', K32), None, PUR)]),
 ('Same-model judge, anonymised reports', [
   ('Qwen2.5-VL-3B, zero-shot, no facts (n=32)', ('judge_crossfamily_3b_nofacts_notell', K32), None, RED),
   ('3B + adapter, no facts (n=32)', ('judge_crossfamily_lora_nofacts_notell', K32), None, ORA),
   ('Frontier, facts given (n=32)', ('judge_crossfamily_clean_facts_notell', K32), None, PUR),
   ('Frontier, no facts (n=32)', ('judge_crossfamily_clean_nofacts_notell', K32), None, PUR)]),
 ('Same-model judge, no facts, identifying strings present', [
   ('Frontier, image-based (n=32)', ('judge_crossfamily_clean_nofacts', K32), None, PUR),
   ('Frontier, NO image, facts only (n=32)', ('judge_samevendor_full_noimg', K32), None, PUR_L)]),
 ('Cross-vendor judge (Gemini), no facts', [
   ('Frontier, image-based (n=80)', ('judge_crossfamily_gemini_clean_full2', K80), None, PUR),
   ('Frontier, NO image, facts only (n=80)', ('judge_crossfamily_gemini_noimg_full2', K80), None, PUR_L)]),
 ('Low-cost commercial subject (Gemini), judges without the facts', [
   ('Gemini subject, Claude judge (n=32)', ('judge_crossfamily_gemsubj_claude', K32), None, GRN),
   ('Gemini subject, Gemini judge (n=80)', ('judge_crossfamily_gemini_gemsubj_full2', K80), None, GRN_L)])]
J = []
for group, bars in SPEC:
    for label, src, fixed, col in bars:
        k, n, lo, hi = fixed if fixed else tally(*src)
        J.append(dict(group=group, label=label, k=k, n=n, lo=round(lo, 3), hi=round(hi, 3), color=col)); print(f"{group[:28]:28s} | {label:50s} {k}/{n} [{lo:.3f},{hi:.3f}]")
json.dump(J, open(os.path.join(R, 'judging_bars.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1); print('bars:', len(J))
