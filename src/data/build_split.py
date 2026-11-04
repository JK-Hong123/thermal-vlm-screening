# -*- coding: utf-8 -*-
"""
build_split.py — add a frozen, SESSION-level split_group to the manifest.
  - 'train' / 'test' : 80/20 session split of the 14 seen-equipment (Training folder), stratified by
                        (facility x session-danger). Same equipment, different sessions -> within-distribution eval.
  - 'gen_test'       : the 2 unseen equipment (신재생, 접지콘덴서; Validation folder), fully held out -> zero-shot eval.
Sessions never cross groups (no frame leakage). Seed frozen. Saves split_assignment.json.
"""
import json, random, collections, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
MAN = r"<PROJECT_ROOT>\data\manifests\manifest_e.jsonl"
OUT = r"<PROJECT_ROOT>\data\manifests\manifest_e_split.jsonl"
ASSIGN = r"<PROJECT_ROOT>\data\manifests\split_assignment.json"
SEED = 42; TEST_FRAC = 0.20

recs = [json.loads(l) for l in open(MAN, encoding='utf-8')]
# session -> attributes
sess = {}
for r in recs:
    s = r['session']
    d = sess.setdefault(s, {'facility': r['facility'], 'folder': r['split'], 'danger': False})
    if r['img_status'] == 'danger': d['danger'] = True

random.seed(SEED)
group = {}
by_fac = collections.defaultdict(lambda: {'danger': [], 'normal': []})
for s, d in sess.items():
    if d['folder'] == 'val':                       # unseen equipment -> zero-shot held-out
        group[s] = 'gen_test'
    else:
        by_fac[d['facility']]['danger' if d['danger'] else 'normal'].append(s)

for fac, dd in by_fac.items():
    for cls in ('danger', 'normal'):
        lst = sorted(dd[cls]); random.shuffle(lst)
        ntest = round(len(lst) * TEST_FRAC)
        if len(lst) >= 5 and ntest == 0: ntest = 1     # guarantee some test if enough sessions
        for i, s in enumerate(lst):
            group[s] = 'test' if i < ntest else 'train'

# write manifest with split_group
with open(OUT, 'w', encoding='utf-8') as fo:
    for r in recs:
        r['split_group'] = group[r['session']]
        fo.write(json.dumps(r, ensure_ascii=False) + '\n')
json.dump(group, open(ASSIGN, 'w', encoding='utf-8'), ensure_ascii=False)

# report
fr = collections.Counter(); ss = collections.defaultdict(set); dfr = collections.Counter()
for r in recs:
    g = group[r['session']]; fr[g] += 1; ss[g].add(r['session'])
    if r['img_status'] == 'danger': dfr[g] += 1
print("=== split_group 구성 (세션 단위) ===")
for g in ['train', 'test', 'gen_test']:
    print(f"  {g:9s}: {len(ss[g]):3d} 세션 | {fr[g]:6,d} 프레임 | danger {dfr[g]:,} ({100*dfr[g]/max(fr[g],1):.1f}%)")
print("\n=== within-설비 test danger 커버리지 (희소 설비 확인) ===")
fac_test = collections.defaultdict(lambda: {'test': set(), 'test_d': set(), 'train': set()})
for r in recs:
    g = group[r['session']]
    if g in ('train', 'test'):
        fac_test[r['facility']][g].add(r['session'])
        if g == 'test' and r['img_status'] == 'danger': fac_test[r['facility']]['test_d'].add(r['session'])
for fac in sorted(fac_test, key=lambda x: -len(fac_test[x]['train'])):
    d = fac_test[fac]
    print(f"    {fac}: train {len(d['train'])}세션 / test {len(d['test'])}세션 (danger세션 {len(d['test_d'])})")
print(f"\n-> {OUT}\n-> {ASSIGN}")
