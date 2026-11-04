# -*- coding: utf-8 -*-
"""ceiling_ratio.py — does the trained model track a constant fraction of the token-resolution ceiling?
Pairs each experiment's val κ with the analytic ceiling κ at its visual-token grid K (from resolution_ceiling.txt)."""
import json, os, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = r"<PROJECT_ROOT>\results\lora_v2"
ceil = {}
for line in open(r"<PROJECT_ROOT>\results\resolution_ceiling.txt", encoding='utf-8'):
    m = re.match(r'(p\S+ ctx[\d.]+@(\d+))/(K\d+|pixel)\s+([\d.]+)\s+([\d.]+)', line.strip())
    if m: ceil[(m.group(1), m.group(3))] = float(m.group(5))
# experiment -> (ceiling key, K) ; 336px -> 12 merged tokens, 672 -> 24, 1008 -> 36
EXPS = {'A_stretch_smooth': ('p100 ctx2.4@336', 'K12'), 'C_stretch_fullvis': ('p100 ctx2.4@336', 'K12'), 'D_stretch_coord': ('p100 ctx2.4@336', 'K12'),
        'E_v3_672_tight': ('p100 ctx1.4@672', 'K24'), 'E2_v3_1008_tight': ('p100 ctx1.4@1008', 'K36'), 'F_fullvis_coord_672_n6000': ('p100 ctx1.4@672', 'K24')}
print(f"{'experiment':28s} {'K':>4s} {'ceiling κ':>10s} {'val κ':>7s} {'ratio':>6s} | {'lock κ':>7s} {'ratio':>6s}")
for tag, (ck, K) in EXPS.items():
    hp = os.path.join(ROOT, tag, 'history.json')
    if not os.path.exists(hp): continue
    h = json.load(open(hp, encoding='utf-8'))
    if not h['epochs']: continue
    ev = h['epochs'][-1]['eval']; c = ceil.get((ck, K), float('nan'))
    print(f"{tag:28s} {K:>4s} {c:10.3f} {ev['val']['kappa']:7.3f} {ev['val']['kappa']/c:6.2f} | {ev['lock']['kappa']:7.3f} {ev['lock']['kappa']/c:6.2f}")
print("\nceiling = brightest-token-in-box heuristic on the rendered PNG pooled to KxK (upper bound for any model that only sees KxK tokens).")
