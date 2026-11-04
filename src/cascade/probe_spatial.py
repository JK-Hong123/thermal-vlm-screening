# -*- coding: utf-8 -*-
"""
probe_spatial.py — minimal capability probe: can the VLM name which 3x3 cell a marked / bright spot is in?
Three difficulty levels, synthetic images (no thermal ambiguity at all):
  L1 synthetic_blob : black frame + white box + ONE bright blob in a random cell (no marker)
  L2 synthetic_mark : same + cyan circle+cross marker on the blob
  L3 real_mark      : real thermal crop with the cyan marker (= the v2 pass-1 condition)
If the model fails L1/L2 the failure is basic spatial grounding, not thermal interpretation.
usage: python probe_spatial.py [--n 36] [--model ...] [--quant 4bit]
"""
import json, os, sys, io, time, argparse, random
import numpy as np, cv2
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import CASES, parse_csv, clip_roi, MAN, colorize
from protocol_v2 import GRID_KO, GRID_XY, grid_ko
def kappa(pairs):
    if not pairs: return float('nan')
    cats = sorted(set([a for a, _ in pairs] + [b for _, b in pairs])); n = len(pairs)
    po = sum(a == b for a, b in pairs)/n
    pe = sum((sum(a == c for a, _ in pairs)/n)*(sum(b == c for _, b in pairs)/n) for c in cats)
    return (po-pe)/(1-pe) if pe < 1 else float('nan')
OUTD = r"<PROJECT_ROOT>\results\probe"
CYAN = (255, 255, 0)

PROMPT = """이미지에 흰색 사각형(대상 상자)이 있습니다. 상자 내부를 가로 3칸 × 세로 3칸으로 나누었을 때,
{what} 어느 칸에 있습니까?
다음 중 하나만 정확히 출력하십시오(다른 말 금지):
상단 좌측 / 상단 중앙 / 상단 우측 / 중단 좌측 / 중앙 / 중단 우측 / 하단 좌측 / 하단 중앙 / 하단 우측"""

def synth(cell_idx, marker, size=336, box=(60, 60, 276, 276)):
    img = np.zeros((size, size, 3), np.uint8)
    x1, y1, x2, y2 = box
    gx, gy = cell_idx % 3, cell_idx // 3
    w, h = (x2-x1)/3, (y2-y1)/3
    cx, cy = int(x1 + (gx+0.5)*w), int(y1 + (gy+0.5)*h)
    blob = np.zeros((size, size), np.float32)
    cv2.circle(blob, (cx, cy), int(min(w, h)*0.32), 1.0, -1)
    blob = cv2.GaussianBlur(blob, (31, 31), 0)
    img = cv2.applyColorMap((np.clip(blob*0.9+0.05, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    if marker:
        cv2.circle(img, (cx, cy), 9, CYAN, 2); cv2.drawMarker(img, (cx, cy), CYAN, cv2.MARKER_CROSS, 22, 2)
    return img

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=36)
    ap.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct'); ap.add_argument('--quant', default=None)
    a = ap.parse_args(); os.makedirs(OUTD, exist_ok=True); random.seed(0)
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    kw = dict(device_map='auto')
    if a.quant:
        from transformers import BitsAndBytesConfig
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type='nf4')
    else: kw['torch_dtype'] = torch.bfloat16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, **kw); proc = AutoProcessor.from_pretrained(a.model)
    def ask(path, what):
        msgs = [{"role": "user", "content": [{"type": "image", "image": path}, {"type": "text", "text": PROMPT.format(what=what)}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ii, vi = process_vision_info(msgs)
        inp = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad(): g = model.generate(**inp, max_new_tokens=16, do_sample=False)
        return proc.batch_decode(g[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    def norm(ans):
        for c in sorted(GRID_KO, key=len, reverse=True):   # longest first ('중앙' ⊂ '하단 중앙')
            if c in ans: return c
        return '파싱실패'
    results = {}
    # L1 / L2 synthetic
    for level, marker in [('L1_synth_blob', False), ('L2_synth_mark', True)]:
        pairs = []
        for k in range(a.n):
            ci = k % 9
            fp = os.path.join(OUTD, f"{level}_{k:02d}.png"); cv2.imwrite(fp, synth(ci, marker))
            what = '하늘색 원+십자 표식이' if marker else '가장 밝은(고온) 지점이'
            pairs.append((norm(ask(fp, what)), GRID_KO[ci]))
        acc = sum(p == t for p, t in pairs)/len(pairs); k_ = kappa([(p, t) for p, t in pairs if p in GRID_XY])
        results[level] = dict(n=len(pairs), acc=round(acc, 3), kappa=round(k_, 3),
                              answers={c: sum(1 for p, _ in pairs if p == c) for c in GRID_KO if any(p == c for p, _ in pairs)},
                              parse_fail=sum(1 for p, _ in pairs if p == '파싱실패'))
        print(level, results[level])
    # L3 real thermal crops (same marker as v2 pass-1)
    idx = [json.loads(l) for l in open(os.path.join(CASES, 'index.jsonl'), encoding='utf-8')][:a.n]
    pairs = []
    for it in idx:
        f = json.load(open(os.path.join(CASES, it['case_id'], 'facts.json'), encoding='utf-8'))
        fp = os.path.join(CASES, it['case_id'], 'thermal_crop_v2.png')
        pairs.append((norm(ask(fp, '하늘색 원+십자 표식이')), grid_ko(f['thermal']['hotspot_position_in_target'])))
    acc = sum(p == t for p, t in pairs)/len(pairs); k_ = kappa([(p, t) for p, t in pairs if p in GRID_XY])
    results['L3_real_mark'] = dict(n=len(pairs), acc=round(acc, 3), kappa=round(k_, 3),
                                   answers={c: sum(1 for p, _ in pairs if p == c) for c in GRID_KO if any(p == c for p, _ in pairs)},
                                   parse_fail=sum(1 for p, _ in pairs if p == '파싱실패'))
    print('L3_real_mark', results['L3_real_mark'])
    results['_model'] = a.model + ('/' + a.quant if a.quant else '')
    tag = 'probe_' + a.model.split('/')[-1].replace('.', '') + ('_' + a.quant if a.quant else '') + '.json'
    json.dump(results, open(os.path.join(OUTD, tag), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('saved', tag, '| chance level = 0.111')

if __name__ == '__main__':
    main()
