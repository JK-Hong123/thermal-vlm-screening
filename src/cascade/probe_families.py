# -*- coding: utf-8 -*-
"""
probe_families.py — the blind-observation probe (marker cell + marker perturbation) for several small VLM families,
so the 'small VLMs cannot ground thermal images' claim is not tied to one family or to the Korean prompt.
Generic loader: AutoModelForImageTextToText + AutoProcessor (transformers >= 5), chat template with images.
usage: python probe_families.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct --lang en [--quant 4bit] [--n 80]
outputs: results/vlm_out/families/<model_tag>_<lang>.json (+ per-case raw)
"""
import json, os, sys, io, re, time, argparse, collections
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import MAN, parse_csv, clip_roi
import protocol_v2 as P
L = r"<PROJECT_ROOT>\results\cases_lock"; OUT = r"<PROJECT_ROOT>\results\vlm_out\families"
GRID_KO = P.GRID_KO
GRID_EN = ['top-left', 'top-center', 'top-right', 'middle-left', 'center', 'middle-right', 'bottom-left', 'bottom-center', 'bottom-right']
PROMPT_KO = """이미지 2에 흰색 사각형(대상 상자)이 있고, 하늘색 원+십자 표식(마커)이 그 안에 있습니다. 상자 내부를 가로 3칸 × 세로 3칸으로 나누었을 때 마커가 어느 칸에 있습니까?
다음 중 하나만 정확히 출력하십시오(다른 말 금지):
상단 좌측 / 상단 중앙 / 상단 우측 / 중단 좌측 / 중앙 / 중단 우측 / 하단 좌측 / 하단 중앙 / 하단 우측"""
PROMPT_EN = """Image 2 shows a white rectangle (the target box) with a cyan circle-and-cross marker inside it. Divide the inside of the white box into a 3x3 grid. Which cell contains the marker?
Answer with exactly one of these and nothing else:
top-left / top-center / top-right / middle-left / center / middle-right / bottom-left / bottom-center / bottom-right"""

def kappa(pairs):
    if not pairs: return float('nan')
    cats = sorted(set([a for a, _ in pairs] + [b for _, b in pairs])); n = len(pairs); po = sum(a == b for a, b in pairs)/n
    pe = sum((sum(a == c for a, _ in pairs)/n)*(sum(b == c for _, b in pairs)/n) for c in cats); return (po-pe)/(1-pe) if pe < 1 else float('nan')
def parse_cell(o, lang):
    o = o.strip().lower()
    if lang == 'en':
        for i, c in enumerate(GRID_EN):
            if c in o: return GRID_KO[i]
        v = 'top' if 'top' in o else ('bottom' if 'bottom' in o else ('middle' if 'middle' in o or 'center' in o else None))
        h = 'left' if 'left' in o else ('right' if 'right' in o else ('center' if 'center' in o or 'middle' in o else None))
        if v and h:
            name = 'center' if (v == 'middle' and h == 'center') else f"{v}-{h}"
            return GRID_KO[GRID_EN.index(name)] if name in GRID_EN else '파싱실패'
        return '파싱실패'
    for c in sorted(GRID_KO, key=len, reverse=True):   # longest first ('중앙' ⊂ '하단 중앙')
        if c in o: return c
    v = '상단' if ('상' in o and '하' not in o) else ('하단' if '하' in o else ('중단' if '중' in o else None))
    h = '좌측' if '좌' in o else ('우측' if '우' in o else ('중앙' if '중앙' in o else None))
    p = '중앙' if (v == '중단' and h == '중앙') else (f'{v} {h}' if v and h else '파싱실패')
    return p if p in GRID_KO else '파싱실패'
def perturbed_point(b, px, py):
    w, h = b['width'], b['height']; gx = min(2, int(3*px/max(w, 1))); gy = min(2, int(3*py/max(h, 1))); nx, ny = 2-gx, 2-gy
    if (nx, ny) == (gx, gy): nx = (gx+1) % 3
    return int(b['x'] + (nx+0.5)*w/3), int(b['y'] + (ny+0.5)*h/3)
def cell_of(b, x, y):
    gx = min(2, int(3*(x-b['x'])/max(b['width'], 1))); gy = min(2, int(3*(y-b['y'])/max(b['height'], 1))); return GRID_KO[gy*3+gx]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--lang', default='en', choices=['en', 'ko'])
    ap.add_argument('--quant', default=None); ap.add_argument('--n', type=int, default=80); ap.add_argument('--max_new', type=int, default=16)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); tag = a.model.split('/')[-1].replace('.', '_') + '_' + a.lang + ('_' + a.quant if a.quant else '')
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    kw = dict(device_map='cuda:0')
    if a.quant:
        from transformers import BitsAndBytesConfig
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type='nf4')
    else: kw['torch_dtype'] = torch.bfloat16
    t0 = time.time(); model = AutoModelForImageTextToText.from_pretrained(a.model, **kw); proc = AutoProcessor.from_pretrained(a.model)
    print(f"[{tag}] loaded in {time.time()-t0:.0f}s | params {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    prompt = PROMPT_EN if a.lang == 'en' else PROMPT_KO
    def ask(paths):
        imgs = [Image.open(p).convert('RGB') for p in paths]
        content = []
        for i, _ in enumerate(imgs): content += [{"type": "text", "text": f"Image {i+1}:"}, {"type": "image"}]
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        try:
            inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", images=imgs)
        except TypeError:
            text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            inputs = proc(text=[text], images=imgs, return_tensors="pt")
        inputs = {k: (v.to(model.device) if hasattr(v, 'to') else v) for k, v in inputs.items()}
        if 'pixel_values' in inputs and inputs['pixel_values'].dtype == torch.float32 and not a.quant: inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        with torch.no_grad(): g = model.generate(**inputs, max_new_tokens=a.max_new, do_sample=False)
        return proc.batch_decode(g[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    recs = {json.loads(l)['frame_key']: json.loads(l) for l in open(MAN, encoding='utf-8')}
    idx = [json.loads(l) for l in open(os.path.join(L, 'index.jsonl'), encoding='utf-8')][:a.n]
    rows = []; t0 = time.time()
    for it in idx:
        d = os.path.join(L, it['case_id']); f = json.load(open(os.path.join(d, 'facts.json'), encoding='utf-8')); b = f['bbox']
        truth = P.grid_ko(f['thermal']['hotspot_position_in_target'])
        T = parse_csv(recs[f['frame_key']]['csv']); roi, _ = clip_roi(T, b); py, px = np.unravel_index(int(np.argmax(roi)), roi.shape)
        pert = cell_of(b, *perturbed_point(b, px, py))
        r1 = ask([os.path.join(d, 'thermal_full.png'), os.path.join(d, 'thermal_crop_v2.png')])
        r2 = ask([os.path.join(d, 'thermal_full.png'), os.path.join(d, 'thermal_crop_v2_pert.png')])
        rows.append(dict(case=it['case_id'], truth=truth, pert=pert, raw1=r1, raw2=r2, p1=parse_cell(r1, a.lang), p2=parse_cell(r2, a.lang)))
    n = len(rows); ok1 = [r for r in rows if r['p1'] in GRID_KO]; ok2 = [r for r in rows if r['p2'] in GRID_KO]
    res = dict(model=a.model, lang=a.lang, quant=a.quant, n=n, sec_per_case=round((time.time()-t0)/max(n, 1), 1),
               parse_fail=sum(r['p1'] == '파싱실패' for r in rows),
               exact=round(np.mean([r['p1'] == r['truth'] for r in rows]), 3), kappa=round(kappa([(r['p1'], r['truth']) for r in ok1]), 3),
               follows_moved_marker=round(np.mean([r['p2'] == r['pert'] for r in rows]), 3), kappa_vs_moved=round(kappa([(r['p2'], r['pert']) for r in ok2]), 3),
               same_answer_after_move=round(np.mean([r['p1'] == r['p2'] for r in rows]), 3),
               answer_modes=collections.Counter(r['p1'] for r in rows).most_common(3))
    json.dump({'summary': res, 'rows': rows}, open(os.path.join(OUT, tag + '.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))

if __name__ == '__main__':
    main()
