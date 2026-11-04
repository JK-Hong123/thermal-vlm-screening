# -*- coding: utf-8 -*-
"""
train_cnn_baseline.py — Phase 1 learned baseline.
bbox-level equipment danger classification from thermal-JPG crops. ResNet18 (ImageNet) fine-tuned.
Goal: beat the radiometric-stat ceiling (best single-feature AUROC ~0.78) with a learned model.
"""
import json, random, sys, io, time, collections
import numpy as np, cv2
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import torch, torch.nn as nn
from torchvision import models

MANIFEST = r"<PROJECT_ROOT>\data\manifests\manifest.jsonl"
SZ = 112; N_TRAIN_PER = 15000; N_VAL_PER = 3000
random.seed(0); torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", dev, torch.cuda.get_device_name(0) if dev=='cuda' else '')

def imread_u(p):
    try: return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    except: return None

def auroc(scores, labels):
    s=np.array(scores); y=np.array(labels); order=s.argsort(); ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(1,len(s)+1)
    p=y.sum(); n=len(y)-p
    return float('nan') if p==0 or n==0 else (ranks[y==1].sum()-p*(p+1)/2)/(p*n)

# 1. index instances by split/status
print("indexing manifest...")
insts = {'train':{'danger':[],'normal':[]}, 'val':{'danger':[],'normal':[]}}
for l in open(MANIFEST, encoding='utf-8'):
    r=json.loads(l); sp=r['split']
    if sp not in insts: continue
    for a in r['annotations']:
        st=a.get('status'); b=a.get('bbox')
        if st in ('danger','normal') and b: insts[sp][st].append((r['img'], b))
for sp in insts:
    print(f"  {sp}: danger {len(insts[sp]['danger'])}  normal {len(insts[sp]['normal'])}")

def make_items(sp, n):
    d=random.sample(insts[sp]['danger'], min(n,len(insts[sp]['danger'])))
    no=random.sample(insts[sp]['normal'], min(n,len(insts[sp]['normal'])))
    return [(img,b,1) for img,b in d] + [(img,b,0) for img,b in no]

train_items = make_items('train', N_TRAIN_PER); random.shuffle(train_items)
val_items   = make_items('val', N_VAL_PER); random.shuffle(val_items)
print(f"train {len(train_items)}  val {len(val_items)}")

# 2. precrop into RAM (read each image once)
def load(items, name):
    by_img=collections.defaultdict(list)
    for i,(img,b,lab) in enumerate(items): by_img[img].append((i,b,lab))
    X=np.zeros((len(items),SZ,SZ,3),np.uint8); Y=np.zeros(len(items),np.float32); ok=np.zeros(len(items),bool)
    t0=time.time(); c=0
    for img,lst in by_img.items():
        im=imread_u(img); c+=1
        if im is None: continue
        H,W=im.shape[:2]
        for i,b,lab in lst:
            bw,bh=b['width'],b['height']
            if bw<4 or bh<4: continue
            cx=b['x']+bw/2; cy=b['y']+bh/2; side=max(bw,bh)*2.0  # square context window (2x): equipment + surroundings
            x=int(max(0,cx-side/2)); y=int(max(0,cy-side/2)); x2=int(min(W,cx+side/2)); y2=int(min(H,cy+side/2))
            if x2-x<4 or y2-y<4: continue
            X[i]=cv2.resize(im[y:y2,x:x2],(SZ,SZ)); Y[i]=lab; ok[i]=True
        if c%2000==0: print(f"    {name}: {c} imgs ({time.time()-t0:.0f}s)")
    return X[ok],Y[ok]
import os
cpath = r"<PROJECT_ROOT>\results\crops_ctx.npz"
if os.path.exists(cpath):
    z = np.load(cpath); Xtr,Ytr,Xva,Yva = z['Xtr'],z['Ytr'],z['Xva'],z['Yva']
    print(f"loaded cached crops: train {Xtr.shape} val {Xva.shape}")
else:
    print("pre-cropping (reads each image once)...")
    Xtr,Ytr=load(train_items,'train'); Xva,Yva=load(val_items,'val')
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    np.savez(cpath, Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva)
print(f"cropped: train {len(Ytr)} (danger {int(Ytr.sum())}), val {len(Yva)} (danger {int(Yva.sum())})")

MEAN=np.array([0.485,0.456,0.406],np.float32)*255; STD=np.array([0.229,0.224,0.225],np.float32)*255
def to_tensor(X, train=False):
    x=X.astype(np.float32)
    if train:  # random hflip
        flip=np.random.rand(len(x))<0.5; x[flip]=x[flip,:,::-1,:]
    x=(x-MEAN)/STD
    return torch.from_numpy(x.transpose(0,3,1,2)).contiguous()

# 3. model
net=models.resnet18(weights=None)  # from scratch (offline; weight download was blocked)
net.fc=nn.Linear(512,1); net=net.to(dev)
opt=torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-4)
lossf=nn.BCEWithLogitsLoss()
Yva_t=torch.from_numpy(Yva).to(dev)

def evaluate():
    net.eval(); scores=[]
    with torch.no_grad():
        for i in range(0,len(Xva),256):
            xb=to_tensor(Xva[i:i+256]).to(dev)
            scores.append(torch.sigmoid(net(xb)).squeeze(1).cpu().numpy())
    s=np.concatenate(scores); a=auroc(s,Yva)
    pred=(s>=0.5).astype(int); tp=((pred==1)&(Yva==1)).sum(); fp=((pred==1)&(Yva==0)).sum(); fn=((pred==0)&(Yva==1)).sum()
    prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-9)
    acc=(pred==Yva).mean()
    return a,f1,acc,prec,rec

EPOCHS=12; BS=128
idx=np.arange(len(Ytr)); best_a=0; best_ep=0
for ep in range(EPOCHS):
    net.train(); np.random.shuffle(idx); tl=0; t0=time.time()
    for i in range(0,len(idx),BS):
        bi=idx[i:i+BS]; xb=to_tensor(Xtr[bi],train=True).to(dev); yb=torch.from_numpy(Ytr[bi]).to(dev)
        opt.zero_grad(); out=net(xb).squeeze(1); loss=lossf(out,yb); loss.backward(); opt.step(); tl+=loss.item()
    a,f1,acc,prec,rec=evaluate()
    if a>best_a: best_a=a; best_ep=ep+1; torch.save(net.state_dict(), r"<PROJECT_ROOT>\results\resnet18_danger_best.pt")
    print(f"epoch {ep+1}/{EPOCHS}  loss {tl/(len(idx)//BS):.3f}  |  VAL AUROC {a:.3f}  F1 {f1:.3f}  acc {acc:.3f}  P {prec:.2f} R {rec:.2f}  ({time.time()-t0:.0f}s)")

print("\n=== RESULT ===")
print(f"ResNet18 (context crop, from scratch) BEST VAL AUROC {best_a:.3f} (epoch {best_ep})")
print(f"radiometric-stat ceiling ~0.78  ->  learned model {'BEATS' if best_a>0.78 else 'does NOT beat'} it")
torch.save(net.state_dict(), r"<PROJECT_ROOT>\results\resnet18_danger.pt")
print("saved model.")
