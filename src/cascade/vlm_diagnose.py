# -*- coding: utf-8 -*-
"""
vlm_diagnose.py — Stage-2 grounded diagnosis with Qwen2.5-VL. Prompt template + severity table are files (swappable).
Conditions:
  full        facts + thermal_full + thermal_crop + rgb_crop
  no_rgb      facts + thermal_full + thermal_crop
  facts_only  facts, no images
  image_only  images + equipment name only (no numbers, no gate verdict)  -> hallucination control
  template    deterministic fill of the facts (no VLM)                      -> grounding ceiling control
usage: python vlm_diagnose.py --tag v0 --prompt ..\..\prompts\diagnose_v0.txt --bands ..\..\prompts\severity_bands_v0.json --cond full,facts_only,image_only,template [--cases case000,case001] [--model Qwen/Qwen2.5-VL-3B-Instruct]
"""
import json, os, sys, io, time, argparse, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import CASES
OUT = r"<PROJECT_ROOT>\results\vlm_out"

def band_of(dT, bands):
    for b in bands['bands']:
        if b['dT_min'] <= dT < b['dT_max']: return b
    return bands['bands'][-1]

def band_table(bands):
    return "\n".join(f"- {b['level']} {b['name_ko']}: ΔT {b['dT_min'] if b['dT_min']>-100 else '-∞'} ~ {b['dT_max'] if b['dT_max']<1000 else '∞'}°C → {b['action_ko']}" for b in bands['bands'])

def slots(f, bands, numbers=True):
    t = f['thermal']; g = f['gate']; c = f.get('calib') or {}
    b = band_of(t['dT_scene_max'], bands)
    s = dict(equipment=f['equipment'], component_class=f['component_class'], spec=f['spec'] or '미상', site=f['site'],
             n_components=f['n_components_in_frame'], bbox_position=f['bbox_position_in_frame'],
             t_max=t['t_max'], t_mean=t['t_mean'], scene_median=t['scene_median'], dT_scene_max=t['dT_scene_max'],
             n_siblings=t['n_siblings'], sibling_t_max=t['sibling_t_max'] if t['n_siblings'] else '없음',
             dT_sibling_max=(f"{t['dT_sibling_max']}°C" if t['dT_sibling_max'] is not None else '해당 없음(형제 부품 없음)'),
             ambient=t['ambient_temp'], dT_over_ambient=t['dT_over_ambient'], emissivity=c.get('emissivity'), distance=c.get('obj_distance'),
             hotspot_pos=t['hotspot_position_in_target'], hotspot_area=t['hotspot_area_frac'], n_hotspots=t['n_hotspots'],
             scene_max=t['scene_max'], frame_max_inside='예' if t['frame_max_inside_target'] else '아니오',
             rf_verdict_ko='이상 의심(플래그)' if g['rf_verdict']=='flag' else '정상 범위(통과)', rf_prob=g['rf_prob'],
             rule_verdict_ko='이상' if g['rule_verdict']=='flag' else '정상',
             band_table=band_table(bands), band_level=b['level'], band_name=b['name_ko'], band_action=b['action_ko'])
    if not numbers:
        for k in list(s):
            if k not in ('equipment','component_class','spec','site','n_components','bbox_position','band_table'): s[k] = '(제공되지 않음)'
    return s

def template_fill(f, bands):
    """Deterministic control: report built purely from facts (no model)."""
    t = f['thermal']; g = f['gate']; b = band_of(t['dT_scene_max'], bands)
    flag = g['rf_verdict'] == 'flag'
    sib = f" 동종 부품 대비 온도차는 {t['dT_sibling_max']}°C입니다." if t['dT_sibling_max'] is not None else ""
    return {
        'component': f['equipment'], 'verdict': '이상' if flag else '정상', 'severity': b['level'] if flag else 'L0',
        'thermal_finding': f"대상 최고 온도 {t['t_max']}°C, 장면 중앙값 {t['scene_median']}°C, 장면 대비 온도차 {t['dT_scene_max']}°C. 열점은 대상 내 {t['hotspot_position_in_target']}에 위치.{sib}",
        'evidence_numbers': [f"{t['t_max']}°C", f"{t['scene_median']}°C", f"{t['dT_scene_max']}°C"] + ([f"{t['dT_sibling_max']}°C"] if t['dT_sibling_max'] is not None else []),
        'probable_cause': '확인 불가', 'recommended_action': b['action_ko'] if flag else '정상 감시 유지',
        'uncertainty': f"탐지 모델 이상 확률 {g['rf_prob']}. 단일 프레임 기반이며 부하율·이력은 확인 불가."}

def parse_json(txt):
    m = re.search(r'\{.*\}', txt, re.S)
    if not m: return None
    s = m.group(0)
    for cand in (s, s.replace('\n', ' ')):
        try: return json.loads(cand)
        except Exception: pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--prompt', required=True); ap.add_argument('--bands', required=True); ap.add_argument('--prompt_io', default=None)
    ap.add_argument('--cond', default='full,facts_only,image_only,template'); ap.add_argument('--cases', default='all')
    ap.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct'); ap.add_argument('--max_new', type=int, default=600)
    a = ap.parse_args()
    tmpl = open(a.prompt, encoding='utf-8').read(); bands = json.load(open(a.bands, encoding='utf-8'))
    tmpl_io = open(a.prompt_io, encoding='utf-8').read() if a.prompt_io else tmpl
    idx = [json.loads(l) for l in open(os.path.join(CASES, 'index.jsonl'), encoding='utf-8')]
    if a.cases != 'all': keep = set(a.cases.split(',')); idx = [i for i in idx if i['case_id'] in keep]
    conds = a.cond.split(',')
    need_model = any(c != 'template' for c in conds)
    if need_model:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map='auto')
        proc = AutoProcessor.from_pretrained(a.model)
        def run(content):
            msgs = [{"role": "user", "content": content}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ii, vi = process_vision_info(msgs)
            inp = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to(model.device)
            t0 = time.time()
            with torch.no_grad(): g = model.generate(**inp, max_new_tokens=a.max_new, do_sample=False, repetition_penalty=1.05)
            out = proc.batch_decode(g[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            return out, time.time()-t0, int(inp.input_ids.shape[1]), int(g.shape[1]-inp.input_ids.shape[1])
    for cond in conds:
        od = os.path.join(OUT, a.tag, cond); os.makedirs(od, exist_ok=True)
        print(f"\n=== cond {cond} ({len(idx)} cases) ===")
        for it in idx:
            cid = it['case_id']; d = os.path.join(CASES, cid); f = json.load(open(os.path.join(d, 'facts.json'), encoding='utf-8'))
            of = os.path.join(od, cid + '.json')
            if os.path.exists(of): continue
            if cond == 'template':
                rec = {'case_id': cid, 'cond': cond, 'raw': None, 'parsed': template_fill(f, bands), 'sec': 0, 'in_tok': 0, 'out_tok': 0}
            else:
                s = slots(f, bands, numbers=(cond != 'image_only'))
                prompt = (tmpl_io if cond == 'image_only' else tmpl).format(**s)
                content = []
                if cond in ('full', 'no_rgb', 'image_only'):
                    content.append({"type": "text", "text": "[이미지 1] 고정 온도 스케일(10~80°C, 우측 컬러바) 열화상 전체 화면. 흰색 굵은 박스=대상(TARGET), 회색 박스=동종 부품."})
                    content.append({"type": "image", "image": os.path.join(d, 'thermal_full.png')})
                    content.append({"type": "text", "text": "[이미지 2] 대상 부품 확대 열화상(녹색 십자=최고온 지점)."})
                    content.append({"type": "image", "image": os.path.join(d, 'thermal_crop.png')})
                    if cond != 'no_rgb' and f['images'].get('rgb_crop'):
                        content.append({"type": "text", "text": "[이미지 3] 같은 위치의 가시광 사진(참고용, 열화상과 픽셀 정렬은 근사)."})
                        content.append({"type": "image", "image": os.path.join(d, 'rgb_crop.png')})
                content.append({"type": "text", "text": prompt})
                raw, sec, itok, otok = run(content)
                rec = {'case_id': cid, 'cond': cond, 'raw': raw, 'parsed': parse_json(raw), 'sec': round(sec, 2), 'in_tok': itok, 'out_tok': otok}
            json.dump(rec, open(of, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            p = rec['parsed'] or {}
            print(f"  {cid} [{it['stratum'][:14]:14s}] {rec['sec']:5.1f}s json={'ok' if rec['parsed'] else 'FAIL'} verdict={p.get('verdict')} sev={p.get('severity')}")

if __name__ == '__main__':
    main()
