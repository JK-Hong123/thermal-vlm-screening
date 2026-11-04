# -*- coding: utf-8 -*-
"""
build_manifest.py — Phase 1 data pipeline.
Match AI-Hub #117 industrial-thermal labels (JSON) to source files (thermal JPG, RGB JPG, radiometric CSV),
and emit a clean manifest with normal/danger labels for the danger-recognition study.

Output: data/manifests/manifest.jsonl  (one row per labeled image)
Also prints dataset statistics (normal vs danger, per-equipment, per-split, match rate).
"""
import os, json, sys, io, re, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

DATA = r"F:\열화상데이터_5training+1test\117.산업시설 열화상 CCTV 데이터\01.데이터"
SPLITS = {'train': '1.Training', 'val': '2.Validation'}
OUT = r"<PROJECT_ROOT>\data\manifests\manifest.jsonl"

def derive(json_dir, name):
    d = json_dir.replace('라벨링데이터', '원천데이터')
    d = re.sub(r'([\\/])TL(\d)', r'\1TS\2', d)
    cands = [os.path.join(d, name)]
    dc = re.sub(r'([\\/])TS(\d)\1TS\2', r'\1TS\2', d)      # collapse doubled TSn\TSn
    cands.append(os.path.join(dc, name))
    for c in cands:
        if os.path.exists(c): return c
    return None

os.makedirs(os.path.dirname(OUT), exist_ok=True)
t0 = time.time()
n_json = n_ok = n_miss_csv = n_miss_img = 0
img_status = collections.Counter(); ann_status = collections.Counter()
per_equip = collections.Counter(); per_equip_danger = collections.Counter()
per_split = collections.Counter(); dims = collections.Counter()
miss_examples = []

with open(OUT, 'w', encoding='utf-8') as fo:
    for split, sub in SPLITS.items():
        labelroot = os.path.join(DATA, sub, '라벨링데이터')
        for dp, dn, fn in os.walk(labelroot):
            for f in fn:
                if not f.endswith('.json'): continue
                jp = os.path.join(dp, f); n_json += 1
                try: o = json.load(open(jp, encoding='utf-8'))
                except: continue
                md = o.get('metadata', {})
                img_name = o.get('image', {}).get('filename', '')
                rgb_name = o.get('image', {}).get('filename_rgb', '')
                csv_name = o.get('csv', {}).get('filename_csv', '')
                img_p = derive(dp, img_name) if img_name else None
                rgb_p = derive(dp, rgb_name) if rgb_name else None
                csv_p = derive(dp, csv_name) if csv_name else None
                if img_p is None:
                    n_miss_img += 1
                    if len(miss_examples) < 3: miss_examples.append(jp)
                if csv_p is None: n_miss_csv += 1
                if img_p is None: continue
                n_ok += 1
                st = md.get('status'); img_status[st] += 1
                equip = md.get('facility'); per_equip[equip] += 1
                if st == 'danger': per_equip_danger[equip] += 1
                per_split[split] += 1
                dims[(md.get('width'), md.get('height'))] += 1
                anns = []
                for a in o.get('annotations', []):
                    at = a.get('attributes', {})
                    ann_status[at.get('status')] += 1
                    anns.append({'class': a.get('class'), 'status': at.get('status'),
                                 'standard': at.get('standard'), 'bbox': a.get('data')})
                rec = {'split': split, 'status': st, 'facility': equip,
                       'img': img_p, 'rgb': rgb_p, 'csv': csv_p,
                       'w': md.get('width'), 'h': md.get('height'),
                       'emissivity': md.get('emissivity'), 'obj_distance': md.get('obj_distance'),
                       'reflected_temp': md.get('reflected_temp'), 'atmospheric_temp': md.get('atmospheric_temp'),
                       'annotations': anns}
                fo.write(json.dumps(rec, ensure_ascii=False) + '\n')
                if n_json % 20000 == 0:
                    print(f"  ...{n_json} json parsed ({time.time()-t0:.0f}s)")

print(f"\n=== DONE in {time.time()-t0:.0f}s ===")
print(f"json total: {n_json}   matched(img ok): {n_ok}   miss_img: {n_miss_img}   miss_csv: {n_miss_csv}")
print(f"per split: {dict(per_split)}")
print(f"image-level status: {dict(img_status)}")
print(f"annotation-level status: {dict(ann_status)}")
print(f"dims: {dict(dims)}")
print("per-equipment (total / danger):")
for e in per_equip:
    print(f"   {e}: {per_equip[e]}  (danger {per_equip_danger.get(e,0)}, {100*per_equip_danger.get(e,0)/max(per_equip[e],1):.1f}%)")
if miss_examples: print("miss examples:", [os.path.basename(m) for m in miss_examples])
print(f"manifest -> {OUT}")
