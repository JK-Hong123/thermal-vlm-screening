# -*- coding: utf-8 -*-
"""
eval_lora.py — evaluate the spatial LoRA adapter where it matters:
  A) locked test-set cases (test + gen_test sessions, never used for training or prompt tuning)
  B) gen_test UNSEEN equipment only (신재생/접지콘덴서) — does the skill transfer to new equipment?
  C) synthetic probe (out-of-domain images) — did it learn the grid concept or just this dataset?
Reports base vs adapted side by side.
usage: python eval_lora.py --adapter <PROJECT_ROOT>\results\lora_spatial\ep1
"""
import json, os, sys, io, argparse, collections
import numpy as np, cv2
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from common import CASES, MAN, parse_csv, clip_roi, colorize
GRID_KO = ['상단 좌측', '상단 중앙', '상단 우측', '중단 좌측', '중앙', '중단 우측', '하단 좌측', '하단 중앙', '하단 우측']
LOCK = r"<PROJECT_ROOT>\results\cases_lock"
PROBE = r"<PROJECT_ROOT>\results\probe"
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

def cell_ko(b, px, py):
    gx = min(2, int(3*px/max(b['width'], 1))); gy = min(2, int(3*py/max(b['height'], 1)))
    return GRID_KO[gy*3+gx]

def build_lock_items(cases_dir):
    """(image_path, task, truth, equipment, unseen) for every locked case, marked and unmarked."""
    recs = {json.loads(l)['frame_key']: json.loads(l) for l in open(MAN, encoding='utf-8')}
    idx = [json.loads(l) for l in open(os.path.join(cases_dir, 'index.jsonl'), encoding='utf-8')]
    items = []
    for it in idx:
        d = os.path.join(cases_dir, it['case_id'])
        f = json.load(open(os.path.join(d, 'facts.json'), encoding='utf-8'))
        b = f['bbox']; T = parse_csv(recs[f['frame_key']]['csv'])
        roi, _ = clip_roi(T, b); py, px = np.unravel_index(int(np.argmax(roi)), roi.shape)
        truth = cell_ko(b, px, py); unseen = f['split_group'] == 'gen_test'
        items.append((os.path.join(d, 'thermal_crop_v2.png'), 'marked', truth, f['equipment'], unseen))
        items.append((os.path.join(d, 'thermal_crop_v2_nomark.png'), 'unmarked', truth, f['equipment'], unseen))
    return items

def build_probe_items(n=36):
    items = []
    for level, marker in (('L1_synth_blob', False), ('L2_synth_mark', True)):
        for k in range(n):
            fp = os.path.join(PROBE, f"{level}_{k:02d}.png")
            if os.path.exists(fp):
                items.append((fp, 'marked' if marker else 'unmarked', GRID_KO[k % 9], level, False))
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', default=r"<PROJECT_ROOT>\results\lora_spatial\ep1")
    ap.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct'); ap.add_argument('--base_too', action='store_true', default=True)
    a = ap.parse_args()
    proc = AutoProcessor.from_pretrained(a.model)
    lock = build_lock_items(LOCK); probe = build_probe_items()
    print(f"locked items {len(lock)} (unseen-equipment {sum(1 for x in lock if x[4])}) | probe items {len(probe)}")

    def run_model(model, items, bs=4):
        preds = []
        model.eval()
        for i in range(0, len(items), bs):
            chunk = items[i:i+bs]; texts, imgs = [], []
            for path, task, *_ in chunk:
                msgs = [{"role": "user", "content": [{"type": "image", "image": path},
                                                     {"type": "text", "text": PROMPT.format(what=WHAT[task])}]}]
                texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
                ii, _ = process_vision_info(msgs); imgs += ii
            old = proc.tokenizer.padding_side; proc.tokenizer.padding_side = 'left'
            b = proc(text=texts, images=imgs, padding=True, return_tensors="pt").to(model.device)
            proc.tokenizer.padding_side = old
            with torch.no_grad(): g = model.generate(**b, max_new_tokens=12, do_sample=False)
            outs = proc.batch_decode(g[:, b['input_ids'].shape[1]:], skip_special_tokens=True)
            for o in outs:
                p = next((c for c in sorted(GRID_KO, key=len, reverse=True) if c in o), None)   # longest first
                if p is None:
                    v = '상단' if ('상' in o and '하' not in o) else ('하단' if '하' in o else ('중단' if '중' in o else None))
                    h = '좌측' if '좌' in o else ('우측' if '우' in o else ('중앙' if '중앙' in o else None))
                    p = '중앙' if (v == '중단' and h == '중앙') else (f'{v} {h}' if v and h else '파싱실패')
                    if p not in GRID_KO: p = '파싱실패'
                preds.append(p)
        return preds

    def score(items, preds, tag):
        res = {}
        def agg(sel, name):
            pr = [(preds[i], items[i][2]) for i in sel]
            if not pr: return
            res[name] = dict(n=len(pr), acc=round(sum(p == t for p, t in pr)/len(pr), 3),
                             kappa=round(kappa([(p, t) for p, t in pr if p in GRID_KO]), 3))
        agg(range(len(items)), 'all')
        for task in ('marked', 'unmarked'): agg([i for i, x in enumerate(items) if x[1] == task], task)
        agg([i for i, x in enumerate(items) if x[4]], 'unseen_equipment')
        agg([i for i, x in enumerate(items) if not x[4]], 'seen_equipment')
        print(f"  [{tag}] " + " | ".join(f"{k}: acc {v['acc']} κ {v['kappa']} (n={v['n']})" for k, v in res.items()))
        return res

    out = {}
    for name, use_adapter in (('base', False), ('lora', True)):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map='cuda:0')
        if use_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.adapter)
        print(f"=== {name} ===")
        out[name] = {'locked': score(lock, run_model(model, lock), f'{name}/locked'),
                     'probe': score(probe, run_model(model, probe), f'{name}/synthetic-probe')}
        del model; torch.cuda.empty_cache()
    json.dump(out, open(r"<PROJECT_ROOT>\results\lora_eval.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print("\nsaved results/lora_eval.json  (chance: acc 0.111, κ 0)")

if __name__ == '__main__':
    main()
