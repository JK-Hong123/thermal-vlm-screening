# -*- coding: utf-8 -*-
"""
build_manifest_e.py — build the working manifest from the migrated E-drive deduped dataset.
Only frames that have a radiometric CSV are included. Source<->label matched by frame_key.
One row per frame (image) with its annotation list + calibration metadata + session key.
Output: <PROJECT_ROOT>\data\manifests\manifest_e.jsonl
"""
import os, re, sys, io, json, collections, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

E = r"<AIHUB_ROOT>\01.데이터"
OUT = r"<PROJECT_ROOT>\data\manifests\manifest_e.jsonl"

def frame_key(n):
    m = re.match(r'^(.*)\d{4}\.[^.]+$', os.path.basename(n)); return m.group(1) if m else None
def sess_key(fk): return re.sub(r'\([^()]*\)$', '', fk) if fk else fk
def ftype(n):  # ('thermal'|'secondary'|'rgb'|'csv'|None)
    m = re.search(r'(\d{2})\d{2}\.(jpg|csv)$', os.path.basename(n))
    if not m: return None
    grp, ext = m.group(1), m.group(2)
    if ext == 'csv': return 'csv'
    return {'01': 'thermal', '02': 'secondary', '03': 'rgb'}.get(grp)

def build_labels(labroot):
    lab = {}
    t0 = time.time(); c = 0
    for dp, dn, fn in os.walk(labroot):
        for f in fn:
            if not f.lower().endswith('.json'): continue
            fk = frame_key(f)
            if not fk: continue
            try: o = json.load(open(os.path.join(dp, f), encoding='utf-8'))
            except: continue
            md = o.get('metadata', {})
            lab[fk] = {
                'region': md.get('region_name'), 'facility': md.get('facility'),
                'img_status': md.get('status'), 'w': md.get('width'), 'h': md.get('height'),
                'calib': {k: md.get(k) for k in ('emissivity','obj_distance','reflected_temp','atmospheric_temp','relative_humidity')},
                'ann': [{'class': a.get('class'), 'status': a.get('attributes',{}).get('status'),
                         'standard': a.get('attributes',{}).get('standard'), 'bbox': a.get('data')}
                        for a in o.get('annotations', [])]
            }
            c += 1
    print(f"  labels {os.path.basename(os.path.dirname(labroot))}: {c} ({time.time()-t0:.0f}s)")
    return lab

def build_source(srcroot):
    src = collections.defaultdict(dict)
    for dp, dn, fn in os.walk(srcroot):
        for f in fn:
            if f.endswith('.zip'): continue
            fk = frame_key(f); t = ftype(f)
            if fk and t and t not in src[fk]:
                src[fk][t] = os.path.join(dp, f)
    return src

os.makedirs(os.path.dirname(OUT), exist_ok=True)
n_written = 0
stats = {'per_split': collections.Counter(), 'img_status': collections.Counter(),
         'ann_status': collections.Counter(), 'equip_sess': collections.defaultdict(set)}
with open(OUT, 'w', encoding='utf-8') as fo:
    for split, tag in [('1.Training', 'train'), ('2.Validation', 'val')]:
        print(f"### {tag} ###")
        lab = build_labels(os.path.join(E, split, '라벨링데이터'))
        src = build_source(os.path.join(E, split, '원천데이터'))
        matched = miss_csv = miss_lab = 0
        for fk, files in src.items():
            if 'csv' not in files: miss_csv += 1; continue          # radiometric required
            if fk not in lab: miss_lab += 1; continue
            L = lab[fk]
            rec = {'split': tag, 'frame_key': fk, 'session': sess_key(fk),
                   'region': L['region'], 'facility': L['facility'], 'img_status': L['img_status'],
                   'w': L['w'], 'h': L['h'], 'calib': L['calib'],
                   'thermal': files.get('thermal'), 'secondary': files.get('secondary'),
                   'rgb': files.get('rgb'), 'csv': files.get('csv'),
                   'annotations': L['ann']}
            fo.write(json.dumps(rec, ensure_ascii=False) + '\n'); n_written += 1; matched += 1
            stats['per_split'][tag] += 1
            stats['img_status'][L['img_status']] += 1
            for a in L['ann']: stats['ann_status'][a['status']] += 1
            stats['equip_sess'][L['facility']].add(sess_key(fk))
        print(f"  matched {matched}, skipped(no-csv {miss_csv}, no-label {miss_lab})")

print(f"\n=== manifest_e.jsonl: {n_written:,} frames ===")
print("per split:", dict(stats['per_split']))
print("image-level status:", dict(stats['img_status']))
print("annotation-level status:", dict(stats['ann_status']))
print("per-equipment sessions:")
for k, v in sorted(stats['equip_sess'].items(), key=lambda x: -len(x[1])):
    print(f"    {k}: {len(v)} sessions")
print(f"-> {OUT}")
