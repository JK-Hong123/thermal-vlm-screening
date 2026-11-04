# -*- coding: utf-8 -*-
"""
protocol_v1.py — code-owned part of the grounded diagnosis protocol.
  * reference-typed severity (sibling ΔT / scene ΔT / normal-population percentile) with cited sources
  * case typing, confidence from session evidence
  * facts block for the VLM prompt, template fallbacks, post-hoc validation, report assembly
Design principles (from v0 failure analysis + adversarial critiques):
  - verdict / severity / every number is CODE-OWNED. The VLM never decides them and may not emit digits.
  - the VLM fills only descriptive fields (part, heat pattern, visual context, cause hypothesis, operator note,
    image-vs-facts consistency), constrained by closed vocabularies where possible; violations -> template fallback.
"""
import re, json
from common import pos_ko

ELEC_CONN = {'계기용변류기(CT)', '계기용변압기(PT)', '계기용변성기(MOF)', '단로기(DS)', '차단기', '전자접촉기',
             '변압기 접속부', '케이블&부스 연결개소', '접지용 콘덴서', '신재생 설비'}
BULK = {'고압전동기', '기어박스', '보일러', '펌프', '차축베어링', '변압기 케이블 연결개소 포함'}

LEVELS = {0: ('L0', '정상 범위', '정상 감시 유지'),
          1: ('L1', '주의', '기록 후 다음 정기점검 시 재확인'),
          2: ('L2', '요주의', '정밀점검 계획 수립(수 주 내), 부하 전류·체결 상태 확인'),
          3: ('L3', '이상', '조속한 정비(가능한 빠른 시기), 부하 저감 검토'),
          4: ('L4', '긴급', '즉시 조치(정지·부하 차단 검토 후 정비)')}

# sources (cited in the report)
SRC_KR = "산업통상자원부 고시 「전기안전관리자의 직무에 관한 고시」 별지 제7호(3상 비교 ΔT ≤5 정상 / 5~10 요주의 / ≥10 이상)"
SRC_NETA_SIM = "ANSI/NETA MTS Table 100.18 유사부품 대비(1~3 조사 / 4~15 시간 허용 시 수리 / >15 즉시 수리)"
SRC_NETA_AMB = "ANSI/NETA MTS Table 100.18 주변 대비(1~10 조사 / 11~20 수리 / 21~40 감시·시정 / >40 즉시)"
SRC_DATA = "학습 데이터 동종 설비 정상 프레임 분포(잠정 기준, 표준 아님)"

def level_sibling(dT):      # 동종 형제 부품 대비 (국내 3상비교법 + NETA 유사부품)
    if dT <= 3: return 0
    if dT <= 5: return 1
    if dT < 10: return 2
    if dT <= 15: return 3
    return 4
def level_scene(dT):        # 장면 중앙값 대비 (주변 대용; NETA 주변대비 준용, 정상 경계는 국내 5°C)
    if dT <= 5: return 0
    if dT <= 10: return 1
    if dT <= 20: return 2
    if dT <= 40: return 3
    return 4
def level_population(pct, dT_pct):   # 상시 고온 설비: 정상 분포 백분위 (잠정)
    if pct is None: return None
    p = max(pct, dT_pct if dT_pct is not None else 0)
    if p < 90: return 0
    if p < 95: return 1
    if p < 99: return 2
    return 3

def severity(f):
    t = f['thermal']; g = f['gate']; n = f.get('norms', {}); eq = f['equipment']
    flagged = g['rf_verdict'] == 'flag'
    out = {}
    if eq in BULK and n.get('equipment_seen_in_training'):
        pct = n.get('t_max_pct_vs_normal'); dpct = n.get('dT_scene_pct_vs_normal')
        lvl = level_population(pct, dpct)
        out.update(reference='population', reference_ko='동종 설비 정상 분포(절대온도) 대비',
                   dT_ref=None, metric_ko=f"대상 최고온 {t['t_max']}°C = 동종 설비 정상 프레임 분포의 상위 {round(100-pct,1)}% (정상 중앙값 {n['normal_t_max_p50']}°C, P95 {n['normal_t_max_p95']}°C)",
                   source=SRC_DATA, level_raw=lvl)
    elif t['dT_sibling_max'] is not None and t['n_siblings'] >= 1:
        lvl = level_sibling(t['dT_sibling_max'])
        out.update(reference='sibling', reference_ko='같은 화면 동종 부품(형제) 최고온 중앙값 대비',
                   dT_ref=t['dT_sibling_max'], metric_ko=f"형제 대비 ΔT {t['dT_sibling_max']}°C (형제 {t['n_siblings']}개, 기준 {t['sibling_reference']}°C)",
                   source=SRC_KR + "; " + SRC_NETA_SIM, level_raw=lvl)
    else:
        lvl = level_scene(t['dT_scene_max'])
        out.update(reference='scene', reference_ko='장면 중앙값(주변 온도 대용, 비표준 기준점) 대비',
                   dT_ref=t['dT_scene_max'], metric_ko=f"장면 대비 ΔT {t['dT_scene_max']}°C (장면 중앙값 {t['scene_median']}°C)",
                   source=SRC_NETA_AMB + " 준용", level_raw=lvl)
    # secondary (always computed for transparency)
    out['scene_level'] = level_scene(t['dT_scene_max'])
    # final level: gate decides existence of a finding; severity only when flagged (min L1)
    if flagged:
        base = out['level_raw'] if out['level_raw'] is not None else out['scene_level']
        if out['reference'] == 'population':   # normally-hot equipment: absolute-temperature population is wide -> floor with scene dT one notch down
            base = max(base, out['scene_level'] - 1)
            out['metric_ko'] += f"; 장면 대비 ΔT {t['dT_scene_max']}°C(상시 고온 설비이므로 한 단계 하향 적용)"
        lvl_final = max(1, base)
    else:
        lvl_final = 0
    out['level'] = lvl_final; out['level_code'], out['level_ko'], out['action_ko'] = LEVELS[lvl_final]
    return out

def case_type(f, sev):
    t = f['thermal']; g = f['gate']; s = f.get('session_agg', {}); flagged = g['rf_verdict'] == 'flag'
    ff = s.get('flag_frac'); nm = s.get('n_frames_component_matched', 0)
    if flagged:
        if nm >= 3 and ff is not None and ff <= 0.34: return 'intermittent', '간헐적 플래그(세션 내 소수 프레임만 이상)'
        if sev['level'] >= 2: return 'clear_overheat', '명확한 과열 소견'
        return 'mild_overheat', '경미한 과열 소견'
    if t['dT_scene_max'] > 5.5:
        return 'hot_but_normal', '장면 대비 고온이나 설비 특성상 정상 범위(게이트 통과)'
    if t['dT_scene_max'] < 4:
        return 'no_thermal_evidence', '단일 프레임에 열적 이상 근거 없음'
    return 'normal', '정상 범위'

def confidence(f, sev):
    s = f.get('session_agg', {}); g = f['gate']; p = g['rf_prob']
    nm = s.get('n_frames_component_matched', 0); ff = s.get('flag_frac')
    flagged = g['rf_verdict'] == 'flag'
    margin = abs(p - 0.5)
    if nm >= 3 and ff is not None:
        consistent = (ff >= 0.67) if flagged else (ff <= 0.33)
        if consistent and margin >= 0.2: return 'high', f"게이트 확률 여유 충분, 세션 {nm}프레임 중 {s['n_flagged']}프레임 플래그로 일관"
        if not consistent: return 'low', f"세션 {nm}프레임 중 {s['n_flagged']}프레임만 플래그(비일관) → 재촬영·부하 확인 권고"
        return 'medium', f"세션 증거는 일관하나 게이트 확률({p})이 경계에 근접"
    if margin >= 0.3: return 'medium', f"단일 프레임 근거(세션 내 동일 부품 매칭 {nm}프레임), 게이트 확률 {p}"
    return 'low', f"단일 프레임 근거이며 게이트 확률({p})이 경계 근접 → 재촬영 권고"

def gate_reasons(f):
    t = f['thermal']; n = f.get('norms', {}); r = []
    if t['dT_scene_max'] > 5.5: r.append('장면 대비 ΔT가 큼')
    if t['dT_sibling_max'] is not None and t['dT_sibling_max'] > 3: r.append('형제 부품 대비 ΔT가 큼')
    if t['hotspot_area_frac'] < 0.08 and t['n_hotspots'] >= 1: r.append('열점이 국부에 집중')
    if n.get('t_max_pct_vs_normal') is not None and n['t_max_pct_vs_normal'] >= 95: r.append('동종 설비 정상 분포 상위 5% 이내')
    if n.get('t_max_pct_vs_normal') is not None and n['t_max_pct_vs_normal'] < 90 and f['gate']['rf_verdict'] == 'pass': r.append('동종 설비 정상 분포 범위 내')
    if t['frame_max_inside_target'] is False and t['dT_scene_max'] > 5.5: r.append('화면 최고온점은 대상 밖에 존재')
    return r or ['특징 조합(형상·온도)에 의한 판정']

CAUSES_ELEC = ['접촉부 접촉저항 증가(체결 이완·산화)', '과부하 또는 상 불평형', '절연 열화', '냉각·통풍 불량', '정상 부하 발열', '확인 불가']
CAUSES_BULK = ['베어링 손상·윤활 불량', '과부하 운전', '냉각 불량', '정상 운전 발열', '확인 불가']
PATTERNS = ['점형(국부 집중)', '면형(전체 고온)', '선형(도체·케이블 방향)', '불명']

def facts_block(f, sev, ctype, conf):
    t = f['thermal']; g = f['gate']; s = f.get('session_agg', {}); n = f.get('norms', {}); c = f.get('calib') or {}
    sib = f"형제 부품 대비 ΔT {t['dT_sibling_max']}°C (형제 {t['n_siblings']}개)" if t['dT_sibling_max'] is not None else "형제 부품 없음(비교 불가)"
    sess = (f"같은 세션 {s['n_frames_in_session']}프레임 중 동일 부품이 매칭된 {s['n_frames_component_matched']}프레임에서 {s['n_flagged']}프레임 플래그(비율 {s['flag_frac']}), 열점 위치 안정성 {s['hotspot_pos_stability']}"
            if s.get('n_frames_component_matched', 0) >= 1 else "세션 내 동일 부품 매칭 없음(단일 프레임 근거)")
    norm = (f"동종 설비 정상 프레임 t_max 중앙값 {n['normal_t_max_p50']}°C, P95 {n['normal_t_max_p95']}°C; 현재 값은 정상 분포의 백분위 {n['t_max_pct_vs_normal']}"
            if n.get('equipment_seen_in_training') else "학습 데이터에 없는 설비(정상 분포 기준 없음)")
    return "\n".join([
        f"[설비] 종류 {f['equipment']} / 사양 {f['spec'] or '미상'} / 현장 {f['site']} / 화면 내 동종 부품 {f['n_components_in_frame']}개, 대상 위치 화면 {f['bbox_position_in_frame']}",
        f"[시스템 판정(확정, 변경 불가)] 게이트: {'이상 의심(플래그)' if g['rf_verdict']=='flag' else '정상 범위(통과)'} — 근거: {', '.join(gate_reasons(f))}",
        f"  심각도 {sev['level_code']} {sev['level_ko']} / 기준: {sev['reference_ko']} / {sev['metric_ko']} / 케이스 유형: {ctype[1]} / 신뢰도 {conf[0]}({conf[1]})",
        f"[열 계측] 대상 최고 {t['t_max']}°C, 평균 {t['t_mean']}°C, 장면 중앙값 {t['scene_median']}°C, 장면 대비 ΔT {t['dT_scene_max']}°C; {sib}; 주변 기온 {t['ambient_temp']}°C, 방사율 {c.get('emissivity')}, 거리 {c.get('obj_distance')}m",
        f"  열점: 대상 내 {pos_ko(t['hotspot_position_in_target'])} (확대 이미지의 녹색 십자), 열점 면적비 {t['hotspot_area_frac']}, 열점 수 {t['n_hotspots']}; 화면 전체 최고온 {t['scene_max']}°C가 대상 내부에 있음: {'예' if t['frame_max_inside_target'] else '아니오'}",
        f"[정상 분포 대비] {norm}",
        f"[세션 증거] {sess}",
    ])

def template_fields(f, sev, ctype):
    t = f['thermal']
    hp = '점형(국부 집중)' if t['hotspot_area_frac'] < 0.08 else ('면형(전체 고온)' if t['hotspot_area_frac'] > 0.3 else '불명')
    note = {'clear_overheat': '현장에서 해당 부위의 체결 상태와 부하 전류를 확인하십시오.',
            'mild_overheat': '다음 점검 시 동일 부위를 재촬영하여 추세를 확인하십시오.',
            'intermittent': '세션 내 판정이 비일관하므로 부하 조건을 기록하며 재촬영하십시오.',
            'hot_but_normal': '설비 특성상 고온이 정상 범위이므로 절대온도 추세만 기록하십시오.',
            'no_thermal_evidence': '단일 프레임에서 열적 근거가 없어 추가 판단은 유보합니다.',
            'normal': '정상 감시를 유지하십시오.'}[ctype[0]]
    return {'component_part': '확인 불가', 'heat_pattern': hp, 'heat_description': '(이미지 서술 없음: 시스템 기본 문구)',
            'visual_context': '확인 불가', 'cause_hypothesis': ['확인 불가'], 'operator_note': note,
            'image_consistent_with_facts': '불명', 'consistency_reason': '(검증 없음)'}

VLM_FIELDS = ['component_part', 'heat_pattern', 'heat_description', 'visual_context', 'cause_hypothesis', 'operator_note',
              'image_consistent_with_facts', 'consistency_reason']
CJK = re.compile(r'[一-鿿぀-ヿ]')

def validate(vlm, f, sev, ctype):
    """returns (fields, violations). Each VLM field is accepted only if it has no digits, no Chinese/Japanese, and respects vocab."""
    tf = template_fields(f, sev, ctype); out = {}; viol = []
    causes = CAUSES_BULK if f['equipment'] in BULK else CAUSES_ELEC
    for k in VLM_FIELDS:
        v = vlm.get(k) if isinstance(vlm, dict) else None
        if v is None: viol.append(f'{k}:missing'); out[k] = tf[k]; continue
        sv = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
        if re.search(r'\d', sv): viol.append(f'{k}:digits'); out[k] = tf[k]; continue
        if CJK.search(sv): viol.append(f'{k}:cjk'); out[k] = tf[k]; continue
        if len(sv) > 220: sv = sv[:220]; viol.append(f'{k}:truncated')
        if k == 'heat_pattern':
            hit = [p for p in PATTERNS if p.split('(')[0] in sv]
            if not hit: viol.append(f'{k}:vocab'); out[k] = tf[k]; continue
            out[k] = hit[0]; continue
        if k == 'cause_hypothesis':
            items = v if isinstance(v, list) else [x.strip() for x in re.split(r'[,;/]', sv)]
            ok = [c for c in causes if any(c.split('(')[0][:4] in str(it) for it in items)]
            if not ok: viol.append(f'{k}:vocab'); out[k] = tf[k]; continue
            out[k] = ok[:2]; continue
        if k == 'image_consistent_with_facts':
            out[k] = '예' if '예' in sv else ('아니오' if '아니' in sv else '불명'); continue
        out[k] = sv.strip()
    return out, viol

def assemble(f, vlm_fields, viol, sev, ctype, conf):
    t = f['thermal']; g = f['gate']
    return {
        'case_id': f['case_id'], 'equipment': f['equipment'], 'site': f['site'], 'spec': f['spec'],
        'verdict': '이상 의심' if g['rf_verdict'] == 'flag' else '정상 범위',
        'severity': f"{sev['level_code']} {sev['level_ko']}", 'severity_level': sev['level'],
        'severity_reference': sev['reference_ko'], 'severity_metric': sev['metric_ko'], 'severity_source': sev['source'],
        'case_type': ctype[0], 'case_type_ko': ctype[1], 'confidence': conf[0], 'confidence_reason': conf[1],
        'measurements': {'t_max': t['t_max'], 'scene_median': t['scene_median'], 'dT_scene': t['dT_scene_max'],
                         'dT_sibling': t['dT_sibling_max'], 'rf_prob': g['rf_prob']},
        'session_evidence': f.get('session_agg', {}),
        'recommended_action': sev['action_ko'] if g['rf_verdict'] == 'flag' else ('절대온도 추세 기록(상시 고온 설비)' if ctype[0] == 'hot_but_normal' else '정상 감시 유지'),
        'limitations': '단일 촬영 기준; 부하율·전류·이력 미확인; ' + ('형제 부품 없음' if t['dT_sibling_max'] is None else '') ,
        'vlm': vlm_fields, 'vlm_violations': viol,
    }

FLIP = {'top':'bottom','bottom':'top','middle':'middle','left':'right','right':'left','center':'center'}
def perturb_facts(f):
    """control: deliberately wrong hotspot position/area in the facts. A VLM that really looks should answer '아니오'."""
    import copy
    g = copy.deepcopy(f); t = g['thermal']; p = t['hotspot_position_in_target']
    if p == 'center': q = 'top-left'
    else:
        v, h = p.split('-'); q = f"{FLIP[v]}-{FLIP[h]}"
        if q == 'middle-center': q = 'top-left'
    t['hotspot_position_in_target'] = q
    t['hotspot_area_frac'] = round(0.6 if t['hotspot_area_frac'] < 0.15 else 0.02, 3)
    return g
