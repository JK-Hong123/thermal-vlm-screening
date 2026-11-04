# -*- coding: utf-8 -*-
"""
train_radiometric_cnn.py — Phase 1 principled model on RADIOMETRIC temperature.
Input = 3-channel context crop from the temperature matrix (CSV), consistently normalized:
  ch0 = absolute temperature (fixed scale, consistent across images)
  ch1 = temperature relative to scene median (context/relative cue)
  ch2 = thermal gradient magnitude (structure)
Small CNN (BN + global pool) to limit overfitting. Honest reporting: fair single-feature baseline
on the SAME val set + CNN final & best val AUROC (best clearly labeled as optimistic).
"""
import json, random, sys, io, time, collections, os
import numpy as np, cv2
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import torch, torch.nn as nn

MANIFEST=r"<PROJECT_ROOT>\data\manifests\manifest.jsonl"
CACHE=r"<PROJECT_ROOT>\results\rad3ch.npz"
SZ=96; N_IMG={'train':6000,'val':2000}
random.seed(0); torch.manual_seed(0)
dev='cuda' if torch.cuda.is_available() else 'cpu'

def parse_csv(cf):
    try: lines=open(cf,encoding='utf-8').read().splitlines()
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
    return float('nan') if p==0 or n==0 else float((r[y==1].sum()-p*(p+1)/2)/(p*n))

def make_3ch(T, b, W, H, scene_med):
    bw,bh=b['width'],b['height']
    cx=b['x']+bw/2; cy=b['y']+bh/2; side=max(bw,bh)*2.0
    x=int(max(0,cx-side/2)); y=int(max(0,cy-side/2)); x2=int(min(W,cx+side/2)); y2=int(min(H,cy+side/2))
    roi=T[y:y2,x:x2]
    if roi.size<16 or x2-x<4 or y2-y<4: return None
    ch0=np.clip((roi-5)/80.0,0,1)                    # absolute temp, fixed scale
    ch1=np.clip((roi-scene_med+15)/40.0,0,1)         # relative to scene median
    gx=cv2.Sobel(roi,cv2.CV_32F,1,0,3); gy=cv2.Sobel(roi,cv2.CV_32F,0,1,3)
    ch2=np.clip(np.sqrt(gx*gx+gy*gy)/50.0,0,1)       # thermal gradient
    img=np.stack([ch0,ch1,ch2],-1)
    return cv2.resize(img,(SZ,SZ))

if os.path.exists(CACHE):
    z=np.load(CACHE); Xtr,Ytr,Xva,Yva,Rtr,Rva = z['Xtr'],z['Ytr'],z['Xva'],z['Yva'],z['Rtr'],z['Rva']
    print(f"loaded cache: train {Xtr.shape} val {Xva.shape}")
else:
    rows=[json.loads(l) for l in open(MANIFEST,encoding='utf-8')]
    bs={'train':{'danger':[],'normal':[]},'val':{'danger':[],'normal':[]}}
    for r in rows:
        if r.get('csv') and r['status'] in('danger','normal') and r['split'] in bs:
            bs[r['split']][r['status']].append(r)
    def collect(sp):
        n=N_IMG[sp]//2
        imgs=random.sample(bs[sp]['danger'],min(n,len(bs[sp]['danger'])))+random.sample(bs[sp]['normal'],min(n,len(bs[sp]['normal'])))
        random.shuffle(imgs); X=[];Y=[];R=[];c=0;t0=time.time()
        for r in imgs:
            T=parse_csv(r['csv']); c+=1
            if T is None: continue
            H,W=T.shape; sm=float(np.median(T))
            for a in r['annotations']:
                st=a.get('status'); b=a.get('bbox')
                if st not in('danger','normal') or not b: continue
                im=make_3ch(T,b,W,H,sm)
                if im is None: continue
                x=max(0,b['x']);y=max(0,b['y']);x2=min(W,x+b['width']);y2=min(H,y+b['height'])
                X.append((im*255).astype(np.uint8)); Y.append(1 if st=='danger' else 0)
                R.append(float(T[y:y2,x:x2].max())-sm)   # single-feature: rise over scene (fair baseline)
            if c%500==0: print(f"  {sp}: {c}/{len(imgs)} ({time.time()-t0:.0f}s), {len(Y)} bboxes")
        return np.array(X,np.uint8),np.array(Y,np.float32),np.array(R,np.float32)
    print("building radiometric 3ch crops (parses CSVs)...")
    Xtr,Ytr,Rtr=collect('train'); Xva,Yva,Rva=collect('val')
    np.savez(CACHE,Xtr=Xtr,Ytr=Ytr,Xva=Xva,Yva=Yva,Rtr=Rtr,Rva=Rva)
print(f"train {len(Ytr)} (danger {int(Ytr.sum())}), val {len(Yva)} (danger {int(Yva.sum())})")

# fair single-feature baseline on THIS val
print(f"\n[fair baseline] single feature 'rise over scene' AUROC on val = {auroc(Rva,Yva):.3f}")

def to_t(X):  return torch.from_numpy((X.astype(np.float32)/255.0).transpose(0,3,1,2)).contiguous()
Xva_t=to_t(Xva).to(dev);
def conv(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.MaxPool2d(2))
net=nn.Sequential(conv(3,16),conv(16,32),conv(32,64),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(0.3),nn.Linear(64,1)).to(dev)
opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4); lossf=nn.BCEWithLogitsLoss()
Xtr_np=Xtr; idx=np.arange(len(Ytr)); best=0; finals=[]
for ep in range(20):
    net.train(); np.random.shuffle(idx)
    for i in range(0,len(idx),128):
        bi=idx[i:i+128]; xb=to_t(Xtr_np[bi]).to(dev)
        if np.random.rand()<0.5: xb=torch.flip(xb,[3])
        yb=torch.from_numpy(Ytr[bi]).to(dev)
        opt.zero_grad(); out=net(xb).squeeze(1); loss=lossf(out,yb); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        s=torch.sigmoid(net(Xva_t).squeeze(1)).cpu().numpy()
    a=auroc(s,Yva); best=max(best,a); finals.append(a)
    if (ep+1)%4==0: print(f"  epoch {ep+1}: val AUROC {a:.3f}")
stable=np.mean(finals[-5:])
print(f"\n=== RESULT: radiometric 3ch CNN ===")
print(f"val AUROC — last5-avg(honest) {stable:.3f} | final {finals[-1]:.3f} | best(optimistic) {best:.3f}")
print(f"fair single-feature baseline (val) = {auroc(Rva,Yva):.3f}")
print(f"honest verdict: CNN {'BEATS' if stable>auroc(Rva,Yva)+0.01 else 'does NOT clearly beat'} the single-feature baseline")
