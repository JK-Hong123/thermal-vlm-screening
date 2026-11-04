# -*- coding: utf-8 -*-
"""
dedup.py — reduce intra-session frame redundancy to free disk space.
Policy: per session, keep up to K evenly-spaced frames of EACH status (danger / normal) separately,
so the rare 'danger' class is well preserved and view diversity is retained. Delete the bulky files
(3 jpg + 1 csv) of dropped samples; JSON labels are kept (tiny, preserves the full record).

  dry-run (default): report kept/dropped, danger preservation, per-equipment/split, estimated space; write droplist.
  --delete         : actually delete files listed in the droplist and report freed bytes.
"""
import json, re, os, sys, io, collections, random, argparse
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

MANIFEST=r"<PROJECT_ROOT>\data\manifests\manifest.jsonl"
DROPLIST=r"<PROJECT_ROOT>\results\droplist.txt"
KEEPMANIFEST=r"<PROJECT_ROOT>\data\manifests\manifest_dedup.jsonl"
K=50
def sess(p):
    b=os.path.basename(p); m=re.match(r'^(.*\(\d+\))\(\d+\)\d+\.jpg$', b); return m.group(1) if m else b
def bulky_files(r):
    img=r['img']; rgb=r.get('rgb'); csv=r.get('csv')
    sec=re.sub(r'0101\.jpg$','0201.jpg',img) if img else None
    return [f for f in [img,sec,rgb,csv] if f]

ap=argparse.ArgumentParser(); ap.add_argument('--delete',action='store_true'); args=ap.parse_args()

if args.delete:
    if not os.path.exists(DROPLIST): print("no droplist; run dry-run first"); sys.exit(1)
    files=[l.strip() for l in open(DROPLIST,encoding='utf-8') if l.strip()]
    freed=0; removed=0; miss=0
    for f in files:
        try:
            sz=os.path.getsize(f); os.remove(f); freed+=sz; removed+=1
        except FileNotFoundError: miss+=1
        except Exception as e:
            if miss<5: print("err",f,e)
            miss+=1
        if removed%50000==0: print(f"  removed {removed} ({freed/1e9:.1f}GB freed)")
    print(f"\nDELETED {removed} files, freed {freed/1e9:.2f} GB  (missing {miss})")
    sys.exit(0)

# ---- dry run ----
rows=[json.loads(l) for l in open(MANIFEST,encoding='utf-8')]
groups=collections.defaultdict(lambda: {'danger':[], 'normal':[]})
for r in rows:
    st=r['status']
    if st in ('danger','normal'): groups[sess(r['img'])][st].append(r)

keep=[]; drop=[]
for s,dd in groups.items():
    for st in ('danger','normal'):
        lst=sorted(dd[st], key=lambda r:r['img'])
        if not lst: continue
        if len(lst)<=K: keep+=lst
        else:
            idx=set(np.linspace(0,len(lst)-1,K).round().astype(int))
            for i,r in enumerate(lst): (keep if i in idx else drop).append(r)

# stats
def eq_ct(recs):
    c=collections.Counter();
    for r in recs: c[r['facility']]+=1
    return dict(c)
kd=sum(1 for r in keep if r['status']=='danger'); kn=len(keep)-kd
dd_=sum(1 for r in drop if r['status']=='danger'); dn=len(drop)-dd_
print(f"sessions {len(groups)}")
print(f"KEEP {len(keep)} samples (danger {kd}, normal {kn})")
print(f"DROP {len(drop)} samples (danger {dd_}, normal {dn})")
print(f"per-equipment KEEP: {eq_ct(keep)}")
print(f"split KEEP: {collections.Counter(r['split'] for r in keep)}")
# estimate freed space from a sample of dropped bulky files
random.seed(0); samp=random.sample(drop, min(2500,len(drop)))
tot=0; cnt=0
for r in samp:
    for f in bulky_files(r):
        try: tot+=os.path.getsize(f); cnt+=1
        except: pass
avg=tot/max(cnt,1)
est=avg*4*len(drop)  # ~4 bulky files/sample
print(f"avg bulky file {avg/1e6:.2f}MB; estimated freed ~{est/1e9:.0f} GB (of ~600GB)")
# write droplist
os.makedirs(os.path.dirname(DROPLIST),exist_ok=True)
n=0
with open(DROPLIST,'w',encoding='utf-8') as fo:
    for r in drop:
        for f in bulky_files(r): fo.write(f+'\n'); n+=1
with open(KEEPMANIFEST,'w',encoding='utf-8') as fo:
    for r in keep: fo.write(json.dumps(r,ensure_ascii=False)+'\n')
print(f"droplist written ({n} files) -> {DROPLIST}")
print(f"dedup manifest ({len(keep)} kept) -> {KEEPMANIFEST}")
print("\n>>> DRY RUN ONLY. To delete: run again with --delete")
