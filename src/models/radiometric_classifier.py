# -*- coding: utf-8 -*-
"""
radiometric_classifier.py — Phase 1 learned baseline on RADIOMETRIC temperature (not the JPG).
Insight: thermal JPGs are per-image auto-ranged, so color!=temperature across images -> CNN-on-JPG fails (~0.56).
Radiometric CSV holds absolute temperature -> consistent signal. Train an MLP on per-bbox temperature+context
features and beat the single-feature ceiling (~0.78).
"""
import json, random, sys, io, time, collections
import numpy as np, cv2
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import torch, torch.nn as nn

MANIFEST = r"<PROJECT_ROOT>\data\manifests\manifest.jsonl"
random.seed(0); torch.manual_seed(0)
N_IMG = {'train': 4000, 'val': 1500}   # images per split (balanced by status)
CLASSES = ['고압전동기','차단기','변압기 접속부','단로기(DS)','신재생 설비','접지용 콘덴서']

def parse_csv(cf):
    try: lines = open(cf, encoding='utf-8').read().splitlines()
    except: return None
    rows=[]
    for l in lines[5:]:
        try: r=[float(p) for p in l.split(';') if p.strip()!='']
        except: continue
        if len(r)>=600: rows.append(r[:640])
    return np.array(rows,np.float32) if len(rows)>=400 else None

def auroc(s,y):
    s=np.asarray(s); y=np.asarray(y); o=s.argsort(); r=np.empty_like(o,float); r[o]=np.arange(1,len(s)+1)
    p=y.sum(); n=len(y)-p
    return float('nan') if p==0 or n==0 else (r[y==1].sum()-p*(p+1)/2)/(p*n)

def features(T, anns):
    H,W=T.shape; scene_med=float(np.median(T)); scene_p95=float(np.percentile(T,95)); scene_mean=float(T.mean())
    box_means=[]
    for a in anns:
        b=a.get('bbox')
        if not b: box_means.append(scene_med); continue
        x=max(0,b['x']); y=max(0,b['y']); x2=min(W,x+b['width']); y2=min(H,y+b['height'])
        roi=T[y:y2,x:x2]; box_means.append(float(roi.mean()) if roi.size else scene_med)
    out=[]
    for i,a in enumerate(anns):
        st=a.get('status'); b=a.get('bbox')
        if st not in ('danger','normal') or not b: continue
        x=max(0,b['x']); y=max(0,b['y']); x2=min(W,x+b['width']); y2=min(H,y+b['height'])
        roi=T[y:y2,x:x2]
        if roi.size<16: continue
        gx=cv2.Sobel(roi,cv2.CV_32F,1,0,3); gy=cv2.Sobel(roi,cv2.CV_32F,0,1,3); gmag=np.sqrt(gx*gx+gy*gy)
        others=[box_means[j] for j in range(len(anns)) if j!=i]
        sib_ref=float(np.median(others)) if others else scene_med
        tmax=float(roi.max()); tmean=float(roi.mean()); tp95=float(np.percentile(roi,95)); tstd=float(roi.std())
        feat=[tmax, tmean, tp95, tstd,
              tmax-scene_med, tmean-scene_med, tp95-scene_med,   # rise over scene
              tmax-sib_ref, tmean-sib_ref,                       # contrast vs siblings
              tmax-scene_mean,
              float((roi>scene_p95).mean()),                     # frac hotter than scene p95
              float(gmag.max()), float(gmag.mean()),             # thermal gradient in bbox
              float((x2-x)), float((y2-y)), float((y2-y)/max(x2-x,1))]  # geometry
        cls=a.get('class'); onehot=[1.0 if cls==c else 0.0 for c in CLASSES]
        out.append((feat+onehot, 1 if st=='danger' else 0))
    return out

rows=[json.loads(l) for l in open(MANIFEST,encoding='utf-8')]
bysplit={'train':{'danger':[],'normal':[]},'val':{'danger':[],'normal':[]}}
for r in rows:
    if r.get('csv') and r['status'] in ('danger','normal') and r['split'] in bysplit:
        bysplit[r['split']][r['status']].append(r)

def collect(split):
    n=N_IMG[split]//2
    imgs=random.sample(bysplit[split]['danger'],min(n,len(bysplit[split]['danger']))) + \
         random.sample(bysplit[split]['normal'],min(n,len(bysplit[split]['normal'])))
    random.shuffle(imgs); X=[]; Y=[]; c=0; t0=time.time()
    for r in imgs:
        T=parse_csv(r['csv']); c+=1
        if T is None: continue
        for f,lab in features(T, r['annotations']): X.append(f); Y.append(lab)
        if c%500==0: print(f"  {split}: {c}/{len(imgs)} imgs, {len(Y)} bboxes ({time.time()-t0:.0f}s)")
    return np.array(X,np.float32), np.array(Y,np.float32)

print("collecting radiometric features (parses CSVs)...")
Xtr,Ytr=collect('train'); Xva,Yva=collect('val')
print(f"train bboxes {len(Ytr)} (danger {int(Ytr.sum())}), val {len(Yva)} (danger {int(Yva.sum())})")

# standardize continuous features (first 16 cols) using train stats
mu=Xtr[:,:16].mean(0); sd=Xtr[:,:16].std(0)+1e-6
Xtr[:,:16]=(Xtr[:,:16]-mu)/sd; Xva[:,:16]=(Xva[:,:16]-mu)/sd
Xtr_t=torch.from_numpy(Xtr); Ytr_t=torch.from_numpy(Ytr); Xva_t=torch.from_numpy(Xva)

net=nn.Sequential(nn.Linear(Xtr.shape[1],64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
opt=torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4); lossf=nn.BCEWithLogitsLoss()
best=0
for ep in range(60):
    net.train(); perm=torch.randperm(len(Xtr_t))
    for i in range(0,len(perm),256):
        bi=perm[i:i+256]; opt.zero_grad(); out=net(Xtr_t[bi]).squeeze(1)
        loss=lossf(out,Ytr_t[bi]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): s=torch.sigmoid(net(Xva_t).squeeze(1)).numpy()
    a=auroc(s,Yva); best=max(best,a)
    if (ep+1)%10==0: print(f"  epoch {ep+1}: val AUROC {a:.3f}")
pred=(s>=0.5); tp=((pred==1)&(Yva==1)).sum(); fp=((pred==1)&(Yva==0)).sum(); fn=((pred==0)&(Yva==1)).sum()
prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-9)
print(f"\n=== RESULT: radiometric-feature MLP ===")
print(f"BEST VAL AUROC {best:.3f}   (final F1 {f1:.3f}, P {prec:.2f}, R {rec:.2f})")
print(f"single-feature ceiling ~0.78  ->  {'BEATS' if best>0.78 else 'does NOT beat'} it")
