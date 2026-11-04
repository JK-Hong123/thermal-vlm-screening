# -*- coding: utf-8 -*-
"""
extract_dedup_zip.py — selectively extract a DEDUPLICATED subset directly from the source ZIPs
(no full extraction). Per session: keep ALL danger frames + up to K evenly-spaced normal frames;
extract only those frames' files (csv + 3 jpg). Leaves the ZIPs intact.
"""
import zipfile, os, re, sys, io, collections, json, time
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

VAL = r"F:\열화상데이터_5training+1test\117.산업시설 열화상 CCTV 데이터\01.데이터\2.Validation"
LAB = os.path.join(VAL, '라벨링데이터'); SRC = os.path.join(VAL, '원천데이터')
ZIPS = ['VS2.zip', 'VS3.zip']; K_NORMAL = 50

def frame_key(name):
    b = os.path.basename(name); m = re.match(r'^(.*\(\d+\)\(\d+\))', b); return m.group(1) if m else None
def sess_key(name):
    b = os.path.basename(name); m = re.match(r'^(.*\(\d+\))\(\d+\)', b); return m.group(1) if m else None

# 1. labels -> status per frame
print("indexing val labels for danger preservation...")
status = {}
for dp, dn, fn in os.walk(LAB):
    for f in fn:
        if f.endswith('.json'):
            k = frame_key(f)
            if not k: continue
            try: status[k] = json.load(open(os.path.join(dp, f), encoding='utf-8')).get('metadata', {}).get('status')
            except: pass
print(f"  label frame-keys: {len(status)} (danger {sum(v=='danger' for v in status.values())})")

for zn in ZIPS:
    zp = os.path.join(SRC, zn)
    if not os.path.exists(zp): print(zn, 'missing'); continue
    zf = zipfile.ZipFile(zp)
    members = collections.defaultdict(list); fk_sess = {}
    for i in zf.infolist():
        if i.is_dir(): continue
        k = frame_key(i.filename)
        if not k: continue
        members[k].append(i); fk_sess[k] = sess_key(i.filename)
    by_sess = collections.defaultdict(lambda: {'danger': [], 'normal': []})
    for k in members:
        st = status.get(k, 'normal')
        by_sess[fk_sess[k]]['danger' if st == 'danger' else 'normal'].append(k)
    keep = set()
    for s, dd in by_sess.items():
        keep |= set(dd['danger'])
        nrm = sorted(dd['normal'])
        if len(nrm) <= K_NORMAL: keep |= set(nrm)
        else:
            idx = set(np.linspace(0, len(nrm) - 1, K_NORMAL).round().astype(int))
            keep |= set(nrm[i] for i in idx)
    kept_files = [m for k in keep for m in members[k]]
    ndanger = sum(1 for k in keep if status.get(k) == 'danger')
    print(f"\n{zn}: {len(members)} frames, {len(by_sess)} sessions -> KEEP {len(keep)} frames "
          f"(danger {ndanger}) = {len(kept_files)} files")
    t0 = time.time(); done = 0; b = 0
    for m in kept_files:
        try: zf.extract(m, SRC); b += m.file_size; done += 1
        except Exception as e:
            if done < 3: print("  extract err:", e)
        if done % 1000 == 0: print(f"  ...{done}/{len(kept_files)} ({b/1e9:.1f}GB, {time.time()-t0:.0f}s)")
    print(f"{zn} DONE: extracted {done} files, {b/1e9:.2f} GB in {time.time()-t0:.0f}s")
    zf.close()
print("\nAll done. ZIPs left intact — verify extracted files, then delete VS2.zip/VS3.zip to reclaim ~131GB.")
