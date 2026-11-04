# -*- coding: utf-8 -*-
"""
eval_v3_big.py — independent, larger evaluation of a hotspot-localisation adapter (coord task) on sessions that were
never used for training, prompt tuning, dev cases, or the locked set:
  * within-test  : up to --n bboxes from split_group=='test'     sessions  (seen equipment, new sessions)
  * unseen       : up to --n bboxes from split_group=='gen_test' sessions  (equipment never seen in training)
Renders v3 (672px, ctx1.4, stretch hi=max) on the fly, scores 3x3-cell κ, normalised distance, within-0.2, per equipment.
usage: python eval_v3_big.py --adapter <dir> [--n 300] [--render_only]
"""
import json, os, sys, io, argparse, random, collections, re
import numpy as np, cv2
from multiprocessing import Pool
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import MAN, parse_csv, clip_roi, CASES

def render_v3(T, b, size, ctx):
    H, W = T.shape; bw, bh = b['width'], b['height']; cx = b['x']+bw/2; cy = b['y']+bh/2; s_ = max(bw, bh)*ctx
    x = int(max(0, cx-s_/2)); y = int(max(0, cy-s_/2)); x2 = int(min(W, cx+s_/2)); y2 = int(min(H, cy+s_/2))
    sub = T[y:y2, x:x2]
    if sub.size == 0: return None
    lo, hi = float(np.percentile(sub, 1)), float(sub.max()); hi = max(hi, lo+0.5)
    norm = np.clip((sub-lo)/(hi-lo), 0, 1); col = cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    col = cv2.resize(col, (size, size), interpolation=cv2.INTER_CUBIC); sx, sy = size/(x2-x), size/(y2-y)
    cv2.rectangle(col, (int((b['x']-x)*sx), int((b['y']-y)*sy)), (int((b['x']-x+bw)*sx), int((b['y']-y+bh)*sy)), (255, 255, 255), 3)
    return col
GRID_KO = ['상단 좌측', '상단 중앙', '상단 우측', '중단 좌측', '중앙', '중단 우측', '하단 좌측', '하단 중앙', '하단 우측']
OUTD = r"<PROJECT_ROOT>\results\eval_v3_big"; LOCK = r"<PROJECT_ROOT>\results\cases_lock"
PROMPT = """이미지에 흰색 사각형(대상 상자)이 있습니다. 상자의 왼쪽 위를 (0.00,0.00), 오른쪽 아래를 (1.00,1.00)이라 할 때,
가장 밝은(고온) 지점의 좌표를 "x,y" 형식(소수 둘째 자리)으로만 출력하십시오."""

def kappa(pairs):
    if not pairs: return float('nan')
    cats = sorted(set([a for a, _ in pairs] + [b for _, b in pairs])); n = len(pairs)
    po = sum(a == b for a, b in pairs)/n
    pe = sum((sum(a == c for a, _ in pairs)/n)*(sum(b == c for _, b in pairs)/n) for c in cats)
    return (po-pe)/(1-pe) if pe < 1 else float('nan')
def cell_from_xy(x, y): return GRID_KO[min(2, max(0, int(3*y)))*3 + min(2, max(0, int(3*x)))]

def job(args):
    rec, ai, out = args
    T = parse_csv(rec['csv'])
    if T is None: return None
    anns = [a for a in rec['annotations'] if a.get('bbox') and a.get('status') in ('danger', 'normal')]
    b = anns[ai]['bbox']
    if min(b['width'], b['height']) < 24: return None
    roi, _ = clip_roi(T, b)
    if roi.size < 400: return None
    img = render_v3(T, b, 672, 1.4)
    if img is None: return None
    cv2.imwrite(out, img)
    sm = cv2.GaussianBlur(roi, (5, 5), 0); sy, sx = np.unravel_index(int(np.argmax(sm)), sm.shape)
    return dict(image=out, x_norm=float(sx)/max(b['width']-1, 1), y_norm=float(sy)/max(b['height']-1, 1),
                cell=cell_from_xy(float(sx)/max(b['width']-1, 1), float(sy)/max(b['height']-1, 1)),
                equipment=rec['facility'], session=rec['session'], group=rec['split_group'], status=anns[ai]['status'])

def build(n):
    os.makedirs(os.path.join(OUTD, 'img'), exist_ok=True); random.seed(1)
    recs = [json.loads(l) for l in open(MAN, encoding='utf-8')]
    used = set()
    for d in (CASES, LOCK):
        for it in (json.loads(l) for l in open(os.path.join(d, 'index.jsonl'), encoding='utf-8')):
            used.add(json.load(open(os.path.join(d, it['case_id'], 'facts.json'), encoding='utf-8'))['session'])
    jobs = []
    for grp in ('test', 'gen_test'):
        pool = [r for r in recs if r.get('csv') and r['split_group'] == grp and r['session'] not in used]
        by_s = collections.defaultdict(list)
        for r in pool: by_s[r['session']].append(r)
        cand = []
        for s, fr in by_s.items():
            for r in fr[::max(1, len(fr)//3)][:3]:
                anns = [a for a in r['annotations'] if a.get('bbox') and a.get('status') in ('danger', 'normal')]
                for i in range(len(anns)): cand.append((r, i))
        random.shuffle(cand)
        # at most 4 bboxes per session for diversity
        per = collections.Counter(); sel = []
        for r, i in cand:
            if per[r['session']] >= 4: continue
            per[r['session']] += 1; sel.append((r, i))
            if len(sel) >= int(n*1.3): break
        for k, (r, i) in enumerate(sel): jobs.append((r, i, os.path.join(OUTD, 'img', f"{grp}_{k:04d}.png")))
        print(f"{grp}: {len(by_s)} unused sessions, {len(sel)} candidate bboxes")
    with Pool(8) as p: rows = [x for x in p.imap_unordered(job, jobs, chunksize=8) if x]
    out = []
    for grp in ('test', 'gen_test'): out += [r for r in rows if r['group'] == grp][:n]
    with open(os.path.join(OUTD, 'eval.jsonl'), 'w', encoding='utf-8') as fo:
        for r in out: fo.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"eval set: {len(out)} bboxes | " + ' | '.join(f"{g}: {sum(r['group']==g for r in out)} (sessions {len(set(r['session'] for r in out if r['group']==g))})" for g in ('test', 'gen_test')))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--adapter', default=None); ap.add_argument('--n', type=int, default=300); ap.add_argument('--render_only', action='store_true')
    a = ap.parse_args()
    ep = os.path.join(OUTD, 'eval.jsonl')
    rows = [json.loads(l) for l in open(ep, encoding='utf-8')] if os.path.exists(ep) else build(a.n)
    if a.render_only or not a.adapter: return
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    proc = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', torch_dtype=torch.bfloat16, device_map='cuda:0')
    if a.adapter.lower() != 'none': model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    preds = []
    for i in range(0, len(rows), 4):
        ch = rows[i:i+4]; texts, imgs = [], []
        for r in ch:
            msgs = [{"role": "user", "content": [{"type": "image", "image": r['image']}, {"type": "text", "text": PROMPT}]}]
            texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)); ii, _ = process_vision_info(msgs); imgs += ii
        proc.tokenizer.padding_side = 'left'; b = proc(text=texts, images=imgs, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad(): g = model.generate(**b, max_new_tokens=14, do_sample=False)
        for r, o in zip(ch, proc.batch_decode(g[:, b['input_ids'].shape[1]:], skip_special_tokens=True)):
            m = re.search(r'(\d(?:\.\d+)?)\s*,\s*(\d(?:\.\d+)?)', o)
            if m: x, y = float(m.group(1)), float(m.group(2)); preds.append((cell_from_xy(x, y), float(np.hypot(x-r['x_norm'], y-r['y_norm']))))
            else: preds.append(('파싱실패', float('nan')))
    res = {}
    def score(sel, name):
        pr = [(preds[i][0], rows[i]['cell']) for i in sel]; d = [preds[i][1] for i in sel if preds[i][1] == preds[i][1]]
        res[name] = dict(n=len(sel), acc=round(sum(p == t for p, t in pr)/len(pr), 3), kappa=round(kappa([(p, t) for p, t in pr if p in GRID_KO]), 3),
                         mean_dist=round(float(np.mean(d)), 3), within_0_2=round(float(np.mean([x < 0.2 for x in d])), 3), parse_fail=sum(p == '파싱실패' for p, _ in pr))
    score(range(len(rows)), 'all')
    for g in ('test', 'gen_test'): score([i for i, r in enumerate(rows) if r['group'] == g], g)
    for st in ('danger', 'normal'): score([i for i, r in enumerate(rows) if r['status'] == st], 'label=' + st)
    for eq in sorted(set(r['equipment'] for r in rows)):
        sel = [i for i, r in enumerate(rows) if r['equipment'] == eq]
        if len(sel) >= 15: score(sel, 'eq:' + eq)
    tag = os.path.basename(os.path.dirname(a.adapter.rstrip('\\/'))) + '_' + os.path.basename(a.adapter.rstrip('\\/'))
    json.dump(res, open(os.path.join(OUTD, f'result_{tag}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump([dict(session=r['session'], group=r['group'], equipment=r['equipment'], status=r['status'], truth=r['cell'], pred=p[0], dist=p[1]) for r, p in zip(rows, preds)],
              open(os.path.join(OUTD, f'rows_{tag}.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    print(f"\n=== {tag} on independent eval set (chance acc 0.111, κ 0) ===")
    for k, v in res.items(): print(f"  {k:36s} n={v['n']:4d} acc {v['acc']:.3f} κ {v['kappa']:.3f} dist {v['mean_dist']:.3f} within0.2 {v['within_0_2']:.3f} fail {v['parse_fail']}")

if __name__ == '__main__':
    main()
