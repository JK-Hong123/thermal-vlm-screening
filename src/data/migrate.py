# -*- coding: utf-8 -*-
"""
migrate.py — consolidate + dedup the AI-Hub #117 dataset onto E:, clearing F:.
Policy: per session keep <=K evenly-spaced DANGER + <=K NORMAL frames (each frame = its csv+3 jpg).
Danger status comes from the labels. ZIP member names in some label zips are CP949 (fixed on read).

Steps (run separately, verify between):
  --build-index         : read all E label zips -> frame_key: status  (cache). non-destructive.
  --extract ZIP TARGET  : dedup-extract one source zip to TARGET root. non-destructive (no delete).
  --extract ZIP TARGET --delete : same, then delete the zip.
"""
import zipfile, os, re, sys, io, json, collections, pickle, argparse, time
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

E = r"<AIHUB_ROOT>\01.데이터"
TRAIN_LAB = os.path.join(E, '1.Training', '라벨링데이터')
VAL_LAB   = os.path.join(E, '2.Validation', '라벨링데이터')
IDX = r"<PROJECT_ROOT>\data\manifests\status_index.pkl"
K = 40

def fixname(info):
    if info.flag_bits & 0x800: return info.filename          # UTF-8 flag set
    try: return info.filename.encode('cp437').decode('cp949') # recover Korean
    except: return info.filename
def frame_key(name):
    # strip trailing 4-digit type code + extension (e.g. '0101.jpg','0102.csv','0301.jpg')
    m = re.match(r'^(.*)\d{4}\.[^.]+$', os.path.basename(name)); return m.group(1) if m else None
def sess_key(fk):
    # session = frame id minus the trailing '(frame-index)' group; keeps any side marker like 'R'
    return re.sub(r'\([^()]*\)$', '', fk) if fk else fk

def build_index():
    idx = {}
    for labdir in [TRAIN_LAB, VAL_LAB]:
        for z in sorted(os.listdir(labdir)):
            if not z.endswith('.zip'): continue
            zf = zipfile.ZipFile(os.path.join(labdir, z)); t0 = time.time(); c = 0
            for info in zf.infolist():
                if info.is_dir() or not info.filename.lower().endswith('.json'): continue
                fk = frame_key(info.filename)
                if not fk: continue
                try:
                    idx[fk] = json.loads(zf.read(info)).get('metadata', {}).get('status'); c += 1
                except: pass
            zf.close(); print(f"  {z}: +{c} ({time.time()-t0:.0f}s) total {len(idx)}")
    os.makedirs(os.path.dirname(IDX), exist_ok=True)
    pickle.dump(idx, open(IDX, 'wb'))
    print(f"INDEX DONE: {len(idx)} frames, danger {sum(v=='danger' for v in idx.values())}")

def dedup_extract(zip_path, target_root, idx, delete=False):
    zf = zipfile.ZipFile(zip_path)
    members = collections.defaultdict(list); fk_sess = {}
    for info in zf.infolist():
        if info.is_dir(): continue
        fk = frame_key(info.filename)
        if not fk: continue
        members[fk].append(info); fk_sess[fk] = sess_key(fk)
    by_sess = collections.defaultdict(lambda: {'danger': [], 'normal': []})
    for fk in members:
        by_sess[fk_sess[fk]]['danger' if idx.get(fk) == 'danger' else 'normal'].append(fk)
    keep = set()
    for s, dd in by_sess.items():
        for cls in ('danger', 'normal'):
            lst = sorted(dd[cls])
            if len(lst) <= K: keep |= set(lst)
            else:
                ii = set(np.linspace(0, len(lst) - 1, K).round().astype(int))
                keep |= set(lst[i] for i in ii)
    kept = [m for fk in keep for m in members[fk]]
    kd = sum(1 for fk in keep if idx.get(fk) == 'danger')
    print(f"  {os.path.basename(zip_path)}: {len(members)} frames, {len(by_sess)} sessions -> "
          f"keep {len(keep)} (danger {kd}) = {len(kept)} files")
    t0 = time.time(); n = 0; b = 0
    for info in kept:
        out = os.path.join(target_root, fixname(info).replace('/', os.sep))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'wb') as f: f.write(zf.read(info))
        n += 1; b += info.file_size
        if n % 3000 == 0: print(f"    ...{n}/{len(kept)} ({b/1e9:.1f}GB, {time.time()-t0:.0f}s)")
    zf.close()
    print(f"  EXTRACTED {n} files, {b/1e9:.2f}GB in {time.time()-t0:.0f}s")
    if delete:
        os.remove(zip_path); print(f"  DELETED {os.path.basename(zip_path)}")

def extract_all_labels(lab_dir, delete=False):
    """extract ALL json labels (fixed names) from every *.zip in lab_dir, keep all (tiny), then delete zip."""
    for z in sorted(f for f in os.listdir(lab_dir) if f.endswith('.zip')):
        zp = os.path.join(lab_dir, z); zf = zipfile.ZipFile(zp); n = 0; t0 = time.time()
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith('.json'): continue
            out = os.path.join(lab_dir, fixname(info).replace('/', os.sep))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, 'wb') as f: f.write(zf.read(info)); n += 1
        zf.close(); print(f"  labels {z}: {n} json ({time.time()-t0:.0f}s)")
        if delete: os.remove(zp); print(f"  deleted {z}")

def _present_frames(src_dir):
    present = set()
    for dp, dn, fn in os.walk(src_dir):
        for f in fn:
            if f.lower().endswith(('.jpg', '.csv')):
                fk = frame_key(f)
                if fk: present.add(fk)
    return present

def extract_labels_matched(lab_dir, src_dir, delete=False):
    present = _present_frames(src_dir)
    print(f"  source frames present in {os.path.basename(os.path.dirname(src_dir))}: {len(present)}")
    for z in sorted(f for f in os.listdir(lab_dir) if f.endswith('.zip')):
        zp = os.path.join(lab_dir, z); zf = zipfile.ZipFile(zp); n = 0
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith('.json'): continue
            if frame_key(info.filename) not in present: continue
            out = os.path.join(lab_dir, fixname(info).replace('/', os.sep))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, 'wb') as f: f.write(zf.read(info)); n += 1
        zf.close(); print(f"  labels {z}: extracted {n} matched")
        if delete: os.remove(zp); print(f"  deleted {z}")

def _norm_rel(path, base):
    """relpath minus any leading TSn/TLn wrapper segments -> {region}/{equipment}/file"""
    segs = [s for s in os.path.relpath(path, base).split(os.sep)]
    while segs and re.match(r'^T[LS]\d+$', segs[0], re.I): segs.pop(0)
    return os.sep.join(segs)

def copy_train15():
    """dedup-copy F's TS1..TS5 (source + matched labels) into E train, keep <=K danger + K normal/session."""
    import shutil
    idx = pickle.load(open(IDX, 'rb'))
    F = r"F:\열화상데이터_5training+1test\117.산업시설 열화상 CCTV 데이터\01.데이터\1.Training"
    FTS, FTL = os.path.join(F, '원천데이터'), os.path.join(F, '라벨링데이터')
    E_src = os.path.join(E, '1.Training', '원천데이터'); E_lab = os.path.join(E, '1.Training', '라벨링데이터')
    for n in [1, 2, 3, 4, 5]:
        sdir = os.path.join(FTS, f'TS{n}')
        if not os.path.isdir(sdir): print(f"  TS{n} 없음"); continue
        members = collections.defaultdict(list)
        for dp, dn, fn in os.walk(sdir):
            for f in fn:
                fk = frame_key(f)
                if fk: members[fk].append(os.path.join(dp, f))
        by_sess = collections.defaultdict(lambda: {'danger': [], 'normal': []})
        for fk in members: by_sess[sess_key(fk)]['danger' if idx.get(fk)=='danger' else 'normal'].append(fk)
        keep = set()
        for s, dd in by_sess.items():
            for cls in ('danger', 'normal'):
                lst = sorted(dd[cls])
                keep |= set(lst) if len(lst) <= K else set(lst[i] for i in set(np.linspace(0,len(lst)-1,K).round().astype(int)))
        nf = 0
        for fk in keep:
            for fp in members[fk]:
                out = os.path.join(E_src, _norm_rel(fp, sdir)); os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.copy2(fp, out); nf += 1
        # matching labels from TL{n}
        ldir = os.path.join(FTL, f'TL{n}'); nl = 0
        if os.path.isdir(ldir):
            for dp, dn, fn in os.walk(ldir):
                for f in fn:
                    if f.lower().endswith('.json') and frame_key(f) in keep:
                        out = os.path.join(E_lab, _norm_rel(os.path.join(dp, f), ldir)); os.makedirs(os.path.dirname(out), exist_ok=True)
                        shutil.copy2(os.path.join(dp, f), out); nl += 1
        kd = sum(1 for fk in keep if idx.get(fk)=='danger')
        print(f"  TS{n}: {len(members)} frames, {len(by_sess)} sess -> keep {len(keep)} (danger {kd}) | copied {nf} src + {nl} labels")

def run_source(src_dir, delete=False):
    idx = pickle.load(open(IDX, 'rb'))
    zips = sorted(f for f in os.listdir(src_dir) if f.endswith('.zip'))
    print(f"### source dir {src_dir}: {len(zips)} zips ###")
    for z in zips:
        try:
            print(f"[{z}] {os.path.getsize(os.path.join(src_dir,z))/1e9:.0f}GB")
            dedup_extract(os.path.join(src_dir, z), src_dir, idx, delete=delete)
        except Exception as e:
            print(f"  ERROR on {z}: {e}  (kept zip, continuing)")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-index', action='store_true')
    ap.add_argument('--extract'); ap.add_argument('--target'); ap.add_argument('--delete', action='store_true')
    ap.add_argument('--run-source')          # a source dir: dedup-extract all zips (+ --delete)
    ap.add_argument('--extract-labels')      # a label dir: extract all label zips (+ --delete)
    ap.add_argument('--labels-matched', nargs=2, metavar=('LAB_DIR', 'SRC_DIR'))  # matched labels only
    ap.add_argument('--copy-train15', action='store_true')  # migrate F TS1-5 (source+labels) to E
    a = ap.parse_args()
    if a.copy_train15: copy_train15(); sys.exit(0)
    if a.build_index: build_index()
    elif a.extract: dedup_extract(a.extract, a.target, pickle.load(open(IDX,'rb')), a.delete)
    elif a.run_source: run_source(a.run_source, a.delete)
    elif a.extract_labels: extract_all_labels(a.extract_labels, a.delete)
    elif a.labels_matched: extract_labels_matched(a.labels_matched[0], a.labels_matched[1], a.delete)
