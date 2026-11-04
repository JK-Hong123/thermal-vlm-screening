# -*- coding: utf-8 -*-
"""Trim extracted danger frames to <=50 evenly-spaced per session (policy B).
Deletes only extracted files (ZIPs intact -> recoverable). Normal frames already capped at extraction."""
import os, re, sys, io, json, collections
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

VAL = r"F:\열화상데이터_5training+1test\117.산업시설 열화상 CCTV 데이터\01.데이터\2.Validation"
LAB = os.path.join(VAL, '라벨링데이터')
TARGET = os.path.join(VAL, '원천데이터', '1. 서부발전', '4.신재생 설비')
K_DANGER = 50

def frame_key(name):
    m = re.match(r'^(.*\(\d+\)\(\d+\))', os.path.basename(name)); return m.group(1) if m else None
def sess_of(fk):
    m = re.match(r'^(.*\(\d+\))\(\d+\)$', fk); return m.group(1) if m else fk

print("indexing labels...")
status = {}
for dp, dn, fn in os.walk(LAB):
    for f in fn:
        if f.endswith('.json'):
            k = frame_key(f)
            if k:
                try: status[k] = json.load(open(os.path.join(dp, f), encoding='utf-8')).get('metadata', {}).get('status')
                except: pass

if not os.path.isdir(TARGET): print("target missing:", TARGET); sys.exit(1)
files_by_frame = collections.defaultdict(list)
for f in os.listdir(TARGET):
    k = frame_key(f)
    if k: files_by_frame[k].append(os.path.join(TARGET, f))
print(f"extracted frames on disk: {len(files_by_frame)}")

by_sess = collections.defaultdict(lambda: {'danger': [], 'normal': []})
for k in files_by_frame:
    by_sess[sess_of(k)]['danger' if status.get(k) == 'danger' else 'normal'].append(k)

drop = []
for s, dd in by_sess.items():
    dngr = sorted(dd['danger'])
    if len(dngr) > K_DANGER:
        keepidx = set(np.linspace(0, len(dngr) - 1, K_DANGER).round().astype(int))
        drop += [dngr[i] for i in range(len(dngr)) if i not in keepidx]
kept_d = sum(min(K_DANGER, len(dd['danger'])) for dd in by_sess.values())
kept_n = sum(len(dd['normal']) for dd in by_sess.values())
print(f"sessions {len(by_sess)} | keep danger {kept_d}, normal {kept_n} | drop danger frames {len(drop)}")

freed = 0; n = 0
for k in drop:
    for fp in files_by_frame[k]:
        try: freed += os.path.getsize(fp); os.remove(fp); n += 1
        except: pass
remaining = sum(os.path.getsize(os.path.join(TARGET, f)) for f in os.listdir(TARGET))
print(f"deleted {n} files, freed {freed/1e9:.2f} GB")
print(f"remaining extracted set: {remaining/1e9:.2f} GB")
print("ZIPs intact. Safe to delete VS2.zip/VS3.zip now to reclaim ~131GB.")
