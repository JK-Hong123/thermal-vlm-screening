# -*- coding: utf-8 -*-
"""
train_lora_v2.py — ablation-ready trainer for "find the hotspot from the image alone" (no marker).
Options: --render fixed|stretch, --label argmax|smooth, --targets llm|llm_vismlp|full_vision,
         --task cell|coord (cell = 3x3 name; coord = "x,y" normalized in 0..1 -> scored by cell + distance)
Evaluates on val (held-out train sessions) and on the LOCKED set (test/gen_test sessions) each epoch.
usage: python train_lora_v2.py --tag A --render stretch --label smooth --targets llm_vismlp --n 3000 --epochs 1
"""
import json, os, sys, io, time, argparse, random, collections, re
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import torch
from torch.utils.data import DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from qwen_vl_utils import process_vision_info

DATA = r"<PROJECT_ROOT>\results\lora_data_v2"
LOCK = r"<PROJECT_ROOT>\results\cases_lock"
OUT = r"<PROJECT_ROOT>\results\lora_v2"
GRID_KO = ['상단 좌측', '상단 중앙', '상단 우측', '중단 좌측', '중앙', '중단 우측', '하단 좌측', '하단 중앙', '하단 우측']
PROMPT_CELL = """이미지에 흰색 사각형(대상 상자)이 있습니다. 상자 내부를 가로 3칸 × 세로 3칸으로 나누었을 때,
가장 밝은(고온) 지점이 어느 칸에 있습니까?
다음 중 하나만 정확히 출력하십시오(다른 말 금지):
상단 좌측 / 상단 중앙 / 상단 우측 / 중단 좌측 / 중앙 / 중단 우측 / 하단 좌측 / 하단 중앙 / 하단 우측"""
PROMPT_COORD = """이미지에 흰색 사각형(대상 상자)이 있습니다. 상자의 왼쪽 위를 (0.00,0.00), 오른쪽 아래를 (1.00,1.00)이라 할 때,
가장 밝은(고온) 지점의 좌표를 "x,y" 형식(소수 둘째 자리)으로만 출력하십시오."""
TARGETS = {
    'llm':        ["q_proj", "k_proj", "v_proj", "o_proj"],                                   # LLM attention only
    'llm_vismlp': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # = previous recipe (vision MLP included by name)
    'full_vision': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "qkv", "proj"],  # + vision attention
}

def kappa(pairs):
    if not pairs: return float('nan')
    cats = sorted(set([a for a, _ in pairs] + [b for _, b in pairs])); n = len(pairs)
    po = sum(a == b for a, b in pairs)/n
    pe = sum((sum(a == c for a, _ in pairs)/n)*(sum(b == c for _, b in pairs)/n) for c in cats)
    return (po-pe)/(1-pe) if pe < 1 else float('nan')

def cell_from_xy(x, y):
    gx = min(2, max(0, int(3*x))); gy = min(2, max(0, int(3*y))); return GRID_KO[gy*3+gx]

def answer_of(r, task, label):
    if task == 'cell': return GRID_KO[r['cell_smooth' if label == 'smooth' else 'cell_argmax']]
    return f"{r['x_norm']:.2f},{r['y_norm']:.2f}"

def parse_pred(o, task):
    if task == 'cell':
        p = next((c for c in sorted(GRID_KO, key=len, reverse=True) if c in o), None)   # longest first
        if p is None:
            v = '상단' if ('상' in o and '하' not in o) else ('하단' if '하' in o else ('중단' if '중' in o else None))
            h = '좌측' if '좌' in o else ('우측' if '우' in o else ('중앙' if '중앙' in o else None))
            p = '중앙' if (v == '중단' and h == '중앙') else (f'{v} {h}' if v and h else '파싱실패')
            if p not in GRID_KO: p = '파싱실패'
        return p, None
    m = re.search(r'(\d(?:\.\d+)?)\s*,\s*(\d(?:\.\d+)?)', o)
    if not m: return '파싱실패', None
    x, y = float(m.group(1)), float(m.group(2)); return cell_from_xy(x, y), (x, y)

def lock_rows(render):
    recs = [json.loads(l) for l in open(os.path.join(LOCK, 'index.jsonl'), encoding='utf-8')]; rows = []
    for it in recs:
        d = os.path.join(LOCK, it['case_id']); f = json.load(open(os.path.join(d, 'facts.json'), encoding='utf-8'))
        lab = f.get('hotspot_labels_v2')
        if not lab: continue
        img = os.path.join(d, 'thermal_crop_v2_stretch.png' if render == 'stretch' else 'thermal_crop_v2_nomark.png')
        rows.append(dict(img_fixed=os.path.join(d, 'thermal_crop_v2_nomark.png'), img_stretch=os.path.join(d, 'thermal_crop_v2_stretch.png'),
                         equipment=f['equipment'], unseen=f['split_group'] == 'gen_test', **lab))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--render', default='stretch', choices=['fixed', 'stretch'])
    ap.add_argument('--label', default='smooth', choices=['argmax', 'smooth']); ap.add_argument('--targets', default='llm_vismlp', choices=list(TARGETS))
    ap.add_argument('--task', default='cell', choices=['cell', 'coord']); ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--epochs', type=int, default=1); ap.add_argument('--bs', type=int, default=2); ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--rank', type=int, default=32); ap.add_argument('--eval_n', type=int, default=240)
    ap.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct'); ap.add_argument('--resume', default=None); ap.add_argument('--grad_ckpt', action='store_true')
    ap.add_argument('--img_key', default=None, help='override image field, e.g. img_v3 (lock uses thermal_crop_<key suffix>)'); ap.add_argument('--lock_img', default=None, help='locked-set image filename for --img_key, e.g. thermal_crop_v3_672.png')
    a = ap.parse_args(); od = os.path.join(OUT, a.tag); os.makedirs(od, exist_ok=True)
    torch.manual_seed(0); random.seed(0)
    proc = AutoProcessor.from_pretrained(a.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map='cuda:0')
    model.config.use_cache = False
    if a.grad_ckpt: model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    if a.resume: model = PeftModel.from_pretrained(model, a.resume, is_trainable=True); print('resumed', a.resume)
    else:
        cfg = LoraConfig(r=a.rank, lora_alpha=2*a.rank, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=TARGETS[a.targets])
        model = get_peft_model(model, cfg)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nvis = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'visual' in n)
    print(f"[{a.tag}] render={a.render} label={a.label} targets={a.targets} task={a.task} | trainable {ntr/1e6:.1f}M (vision {nvis/1e6:.1f}M)")
    tr = [json.loads(l) for l in open(os.path.join(DATA, 'train.jsonl'), encoding='utf-8')][:a.n]
    va = [json.loads(l) for l in open(os.path.join(DATA, 'val.jsonl'), encoding='utf-8')][:a.eval_n]
    lk = lock_rows(a.render)
    if a.img_key:
        for r in lk: r[a.img_key] = os.path.join(os.path.dirname(r['img_stretch']), a.lock_img)
    print(f"train {len(tr)} | val {len(va)} | locked {len(lk)} (unseen {sum(r['unseen'] for r in lk)})")
    img_key = a.img_key or ('img_stretch' if a.render == 'stretch' else 'img_fixed'); prompt = PROMPT_COORD if a.task == 'coord' else PROMPT_CELL
    if a.img_key: tr = [r for r in tr if r.get(a.img_key) and os.path.exists(r[a.img_key])]; va = [r for r in va if r.get(a.img_key) and os.path.exists(r[a.img_key])]

    def build(rows, train=True):
        texts, imgs = [], []
        for r in rows:
            msgs = [{"role": "user", "content": [{"type": "image", "image": r[img_key]}, {"type": "text", "text": prompt}]}]
            if train:
                msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer_of(r, a.task, a.label)}]})
                texts.append(proc.apply_chat_template(msgs, tokenize=False))
            else: texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            ii, _ = process_vision_info(msgs); imgs += ii
        old = proc.tokenizer.padding_side; proc.tokenizer.padding_side = 'right' if train else 'left'
        b = proc(text=texts, images=imgs, padding=True, return_tensors="pt"); proc.tokenizer.padding_side = old
        if train:
            labels = b['input_ids'].clone(); labels[labels == proc.tokenizer.pad_token_id] = -100
            for bi, r in enumerate(rows):
                ans = proc.tokenizer(answer_of(r, a.task, a.label), add_special_tokens=False)['input_ids']
                ids = b['input_ids'][bi].tolist(); L = len(ans); pos = -1
                for s_ in range(len(ids)-L, 0, -1):
                    if ids[s_:s_+L] == ans: pos = s_; break
                m = torch.full_like(labels[bi], -100)
                if pos >= 0: m[pos:pos+L] = labels[bi][pos:pos+L]
                labels[bi] = m
            b['labels'] = labels
        return b

    @torch.no_grad()
    def evaluate(rows, name):
        model.eval(); pairs = []; dists = []; sub = collections.defaultdict(list)
        for i in range(0, len(rows), 4):
            ch = rows[i:i+4]; b = build(ch, train=False).to(model.device)
            g = model.generate(**b, max_new_tokens=14, do_sample=False)
            outs = proc.batch_decode(g[:, b['input_ids'].shape[1]:], skip_special_tokens=True)
            for r, o in zip(ch, outs):
                pred, xy = parse_pred(o, a.task); truth = GRID_KO[r['cell_smooth' if a.label == 'smooth' else 'cell_argmax']]
                pairs.append((pred, truth)); sub['unseen' if r.get('unseen') else 'seen'].append((pred, truth))
                if xy: dists.append(float(np.hypot(xy[0]-r['x_norm'], xy[1]-r['y_norm'])))
        model.train()
        res = dict(n=len(pairs), acc=round(sum(p == t for p, t in pairs)/len(pairs), 3), kappa=round(kappa([(p, t) for p, t in pairs if p in GRID_KO]), 3),
                   parse_fail=sum(p == '파싱실패' for p, _ in pairs), modes=collections.Counter(p for p, _ in pairs).most_common(3))
        if dists: res['mean_dist'] = round(float(np.mean(dists)), 3); res['within_0.2'] = round(float(np.mean([d < 0.2 for d in dists])), 3)
        for k, v in sub.items():
            if k == 'unseen' and v: res['unseen_acc'] = round(sum(p == t for p, t in v)/len(v), 3); res['unseen_kappa'] = round(kappa([(p, t) for p, t in v if p in GRID_KO]), 3); res['unseen_n'] = len(v)
        print(f"  [{name}] {res}"); return res

    hist = {'args': vars(a), 'base': {'val': evaluate(va, 'val/base'), 'lock': evaluate(lk, 'lock/base')}, 'epochs': []}
    dl = DataLoader(tr, batch_size=a.bs, shuffle=True, collate_fn=lambda x: x)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    total = max(1, (len(dl)//a.accum)*a.epochs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)
    step = 0; t0 = time.time(); model.train()
    for ep in range(a.epochs):
        run = 0.0; nb = 0
        for bi, rows in enumerate(dl):
            b = build(rows, train=True).to(model.device); out = model(**b); (out.loss/a.accum).backward(); run += out.loss.item(); nb += 1
            if (bi+1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 25 == 0: print(f"  ep{ep+1} step {step}/{total} loss {run/max(nb,1):.4f} ({time.time()-t0:.0f}s)"); run = 0; nb = 0
        ev = {'val': evaluate(va, f'val/ep{ep+1}'), 'lock': evaluate(lk, f'lock/ep{ep+1}')}
        hist['epochs'].append({'epoch': ep+1, 'step': step, 'eval': ev}); model.save_pretrained(os.path.join(od, f'ep{ep+1}'))
        json.dump(hist, open(os.path.join(od, 'history.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"[{a.tag}] done in {time.time()-t0:.0f}s -> {od}")

if __name__ == '__main__':
    main()
