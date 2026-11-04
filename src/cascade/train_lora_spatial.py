# -*- coding: utf-8 -*-
"""
train_lora_spatial.py — LoRA fine-tune Qwen2.5-VL-3B on the free supervision from radiometric CSVs:
"which 3x3 cell of the target box holds the hotspot?" (marked + unmarked variants).
Loss only on the answer tokens. Evaluates cell accuracy / kappa on a held-out-session val set each epoch.
usage: python train_lora_spatial.py [--epochs 2] [--bs 2] [--accum 8] [--lr 1e-4] [--max_steps N] [--vision]
"""
import json, os, sys, io, time, argparse, random, collections
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

DATA = r"<PROJECT_ROOT>\results\lora_data"
OUT = r"<PROJECT_ROOT>\results\lora_spatial"
GRID_KO = ['상단 좌측', '상단 중앙', '상단 우측', '중단 좌측', '중앙', '중단 우측', '하단 좌측', '하단 중앙', '하단 우측']
PROMPT = """이미지에 흰색 사각형(대상 상자)이 있습니다. 상자 내부를 가로 3칸 × 세로 3칸으로 나누었을 때,
{what} 어느 칸에 있습니까?
다음 중 하나만 정확히 출력하십시오(다른 말 금지):
상단 좌측 / 상단 중앙 / 상단 우측 / 중단 좌측 / 중앙 / 중단 우측 / 하단 좌측 / 하단 중앙 / 하단 우측"""
WHAT = {'marked': '하늘색 원+십자 표식이', 'unmarked': '가장 밝은(고온) 지점이'}

def kappa(pairs):
    if not pairs: return float('nan')
    cats = sorted(set([a for a, _ in pairs] + [b for _, b in pairs])); n = len(pairs)
    po = sum(a == b for a, b in pairs)/n
    pe = sum((sum(a == c for a, _ in pairs)/n)*(sum(b == c for _, b in pairs)/n) for c in cats)
    return (po-pe)/(1-pe) if pe < 1 else float('nan')

class DS(Dataset):
    def __init__(self, path): self.rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def build_batch(proc, rows, train=True):
    texts, imgs = [], []
    for r in rows:
        msgs = [{"role": "user", "content": [{"type": "image", "image": r['image']},
                                            {"type": "text", "text": PROMPT.format(what=WHAT[r['task']])}]}]
        if train:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": r['answer']}]})
            texts.append(proc.apply_chat_template(msgs, tokenize=False))
        else:
            texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        ii, _ = process_vision_info(msgs); imgs.append(ii)
    flat = [x for sub in imgs for x in sub]
    old_side = proc.tokenizer.padding_side
    proc.tokenizer.padding_side = 'right' if train else 'left'   # batched generation needs left padding
    batch = proc(text=texts, images=flat, padding=True, return_tensors="pt")
    proc.tokenizer.padding_side = old_side
    if train:
        labels = batch['input_ids'].clone()
        labels[labels == proc.tokenizer.pad_token_id] = -100
        # mask everything except the answer tokens (answer sits right before the final <|im_end|>)
        for bi, r in enumerate(rows):
            ans_ids = proc.tokenizer(r['answer'], add_special_tokens=False)['input_ids']
            ids = batch['input_ids'][bi].tolist(); L = len(ans_ids); pos = -1
            for s in range(len(ids)-L, 0, -1):
                if ids[s:s+L] == ans_ids: pos = s; break
            mask = torch.full_like(labels[bi], -100)
            if pos >= 0: mask[pos:pos+L] = labels[bi][pos:pos+L]
            labels[bi] = mask
        batch['labels'] = labels
    return batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=2); ap.add_argument('--bs', type=int, default=2); ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--rank', type=int, default=32)
    ap.add_argument('--max_steps', type=int, default=0); ap.add_argument('--vision', action='store_true', help='also adapt the vision tower')
    ap.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct'); ap.add_argument('--eval_n', type=int, default=200); ap.add_argument('--resume', default=None); ap.add_argument('--ep_offset', type=int, default=0)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0); random.seed(0)
    proc = AutoProcessor.from_pretrained(a.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map='cuda:0')
    model.config.use_cache = False
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    cfg = LoraConfig(r=a.rank, lora_alpha=2*a.rank, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                     target_modules=targets,
                     modules_to_save=None)
    if not a.vision:   # restrict LoRA to the language model, leave the vision tower untouched
        cfg.target_modules = targets
        cfg.layers_to_transform = None
    if a.resume:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.resume, is_trainable=True); print('resumed from', a.resume)
    else:
        model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable/1e6:.1f}M | vision adapted: {a.vision}")
    tr = DS(os.path.join(DATA, 'train.jsonl')); va = DS(os.path.join(DATA, 'val.jsonl'))
    print(f"train {len(tr)} | val {len(va)}")
    dl = DataLoader(tr, batch_size=a.bs, shuffle=True, collate_fn=lambda x: x)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    total = (len(dl)//a.accum)*a.epochs if not a.max_steps else a.max_steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(total, 1), pct_start=0.1)

    @torch.no_grad()
    def evaluate(n):
        model.eval(); rows = va.rows[:n]; pairs = collections.defaultdict(list)
        for i in range(0, len(rows), 4):
            chunk = rows[i:i+4]
            b = build_batch(proc, chunk, train=False).to(model.device)
            g = model.generate(**b, max_new_tokens=12, do_sample=False)
            outs = proc.batch_decode(g[:, b['input_ids'].shape[1]:], skip_special_tokens=True)
            for r, o in zip(chunk, outs):
                pred = next((c for c in sorted(GRID_KO, key=len, reverse=True) if c in o), None)   # longest first
                if pred is None:
                    v = '상단' if ('상' in o and '하' not in o) else ('하단' if '하' in o else ('중단' if '중' in o else None))
                    h = '좌측' if '좌' in o else ('우측' if '우' in o else ('중앙' if '중앙' in o else None))
                    pred = ('중앙' if (v == '중단' and h == '중앙') else (f'{v} {h}' if v and h else '파싱실패'))
                    if pred not in GRID_KO and pred != '파싱실패': pred = '파싱실패'
                pairs[r['task']].append((pred, r['answer'])); pairs['all'].append((pred, r['answer']))
        model.train()
        return {k: dict(n=len(v), acc=round(sum(p == t for p, t in v)/len(v), 3),
                        kappa=round(kappa([(p, t) for p, t in v if p in GRID_KO]), 3),
                        modes=collections.Counter(p for p, _ in v).most_common(3)) for k, v in pairs.items()}

    print("=== zero-shot (before training) ===")
    base = evaluate(a.eval_n); [print('  ', k, v) for k, v in base.items()]
    hist = {'base': base, 'epochs': []}
    step = 0; t0 = time.time(); model.train()
    for ep in range(a.epochs):
        run = 0.0; nb = 0
        for bi, rows in enumerate(dl):
            b = build_batch(proc, rows, train=True).to(model.device)
            out = model(**b); loss = out.loss / a.accum
            loss.backward(); run += out.loss.item(); nb += 1
            if (bi+1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 20 == 0:
                    print(f"  ep{ep+1} step {step}/{total} loss {run/max(nb,1):.4f} ({time.time()-t0:.0f}s)"); run = 0; nb = 0
                if a.max_steps and step >= a.max_steps: break
        ev = evaluate(a.eval_n); hist['epochs'].append({'epoch': ep+1+a.ep_offset, 'step': step, 'eval': ev})
        print(f"=== after epoch {ep+1} (step {step}) ==="); [print('  ', k, v) for k, v in ev.items()]
        model.save_pretrained(os.path.join(OUT, f'ep{ep+1+a.ep_offset}'))
        json.dump(hist, open(os.path.join(OUT, f'history_ep{a.ep_offset+1}.json' if a.ep_offset else 'history.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        if a.max_steps and step >= a.max_steps: break
    print("saved adapters to", OUT)

if __name__ == '__main__':
    main()
