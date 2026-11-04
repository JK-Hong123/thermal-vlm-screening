# -*- coding: utf-8 -*-
"""
protocol_v2.py — GTD-v2 (Grounded Thermal Diagnosis): code-owned verdict/severity/numbers +
  pass-1 BLIND observation VLM (images + equipment name only; no hotspot facts) -> code cross-check vs CSV
  pass-2 reasoning VLM on a NUMBER-FREE facts view (band words) + pass-1 observations -> closed-vocab cause/checks/note
  post-hoc validation (digits, CJK, OOV, polarity, cause-compatibility, basis-vs-facts) -> template fallback per field.
Extends protocol_v1 (severity axes S/A/P, case types, confidence) with: review zone, abs-limit flags, sibling contamination,
equipment gate reliability, subtle flag, quality flags, missing list.
"""
import re, json, copy
from common import pos_ko
import protocol_v1 as P1
from protocol_v1 import ELEC_CONN, BULK, LEVELS, level_sibling, level_scene, level_population, CJK

GRID_KO = ['상단 좌측', '상단 중앙', '상단 우측', '중단 좌측', '중앙', '중단 우측', '하단 좌측', '하단 중앙', '하단 우측']
GRID_XY = {'상단 좌측': (0, 0), '상단 중앙': (1, 0), '상단 우측': (2, 0), '중단 좌측': (0, 1), '중앙': (1, 1), '중단 우측': (2, 1),
           '하단 좌측': (0, 2), '하단 중앙': (1, 2), '하단 우측': (2, 2)}
def grid_ko(code):   # 'top-left' -> '상단 좌측'; 'center' -> '중앙'; 'middle-center' -> '중앙'
    k = pos_ko(code)
    return '중앙' if k in ('중앙', '중단 중앙') else k

PARTS = {
 '계기용변류기(CT)': ['일차 단자 접속부', '이차 단자함', '부싱', '본체(절연 몰드)', '지지 구조물'],
 '계기용변압기(PT)': ['일차 단자 접속부', '이차 단자함', '부싱', '본체(절연 몰드)', '지지 구조물'],
 '계기용변성기(MOF)': ['단자 접속부', '부싱', '본체(탱크)', '이차 단자함'],
 '단로기(DS)': ['접촉부(블레이드·클립)', '단자 접속부', '애자', '조작 링크'],
 '차단기': ['상부 단자 접속부', '하부 단자 접속부', '부싱', '소호부·본체', '조작부', '케이블 접속부'],
 '전자접촉기': ['상부 단자', '하부 단자', '코일·본체', '접점부', '보조접점'],
 '변압기 접속부': ['부싱 단자', '리드 접속부', '케이블 접속부', '탱크 표면'],
 '케이블&부스 연결개소': ['부스바 접속부', '케이블 슈 접속부', '볼트 체결부', '케이블 본체'],
 '변압기 케이블 연결개소 포함': ['케이블 접속부', '부싱', '탱크 표면', '방열기'],
 '접지용 콘덴서': ['단자 접속부', '본체', '부싱'],
 '신재생 설비': ['단자·접속부', '본체·모듈', '케이블', '인버터 함체'],
 '고압전동기': ['구동측 베어링 하우징', '반구동측 베어링 하우징', '고정자 프레임', '단자함', '냉각팬 커버'],
 '기어박스': ['입력축 베어링부', '출력축 베어링부', '하우징 본체', '오일 씰부'],
 '펌프': ['베어링 하우징', '메커니컬 씰부', '펌프 케이싱', '커플링부', '전동기측'],
 '보일러': ['노벽·케이싱', '버너부', '배관 접속부', '점검구·도어'],
 '차축베어링': ['차축 베어링(액슬박스)', '휠 림·답면', '브레이크 디스크', '차축 본체'],
}
PATTERNS = ['점형', '면형', '선형', '다점형', '불명']
VISUAL_FLAGS = ['반사 의심', '표면 변색', '이완·부식 흔적', '케이블 손상', '이물질·오염', '특이 없음', '가시광 없음']
QUALITY = ['양호', '보통', '불량']
CAUSES_ELEC = ['접촉부 접촉저항 증가(체결 이완·산화)', '접촉면 오염', '과부하 또는 상 불평형', '절연 열화', '냉각·통풍 불량', '확인 불가']
CAUSES_BULK = ['베어링 손상·윤활 불량', '축 정렬 불량', '씰 마찰·누설', '과부하 운전', '냉각 불량', '확인 불가']
BASIS = ['동종 부품 대비', '열 형태', '육안 특이', '세션 증거', '정상 분포 대비']
CHECKS_ELEC = ['부하 전류·불평형률 측정', '체결 토크 점검', '접촉저항 측정', '접촉면 청소·산화 확인', '절연저항 측정', '냉각·통풍 상태 확인', '각도 바꿔 재촬영(반사 확인)', '부하 조건 기록 후 재촬영']
CHECKS_BULK = ['진동 측정', '윤활 상태·유량 확인', '축 정렬 확인', '냉각·통풍 상태 확인', '부하 조건 기록 후 재촬영', '각도 바꿔 재촬영(반사 확인)']
CHECKS = ['부하 전류·불평형률 측정', '체결 토크 점검', '접촉저항 측정', '접촉면 청소·산화 확인', '절연저항 측정', '냉각·통풍 상태 확인',
          '진동 측정', '윤활 상태·유량 확인', '축 정렬 확인', '각도 바꿔 재촬영(반사 확인)', '부하 조건 기록 후 재촬영']
# cause compatibility: pattern -> allowed causes (None = any)
COMPAT = {'면형': {'과부하 또는 상 불평형', '냉각·통풍 불량', '절연 열화', '과부하 운전', '냉각 불량', '확인 불가'},
          '점형': {'접촉부 접촉저항 증가(체결 이완·산화)', '접촉면 오염', '절연 열화', '베어링 손상·윤활 불량', '씰 마찰·누설', '축 정렬 불량', '확인 불가'},
          '다점형': None, '선형': None, '불명': None}
# absolute-temperature limits (warn, abnormal, basis) — provisional table from Korean 온도패턴법 sources
ABS_LIMITS = {
 '계기용변류기(CT)': (75, 95, 'PT·CT 단자 ≥75 요주의 / 본체 ≥95 이상 (국토안전관리원 온도패턴법)'),
 '계기용변압기(PT)': (75, 95, 'PT·CT 단자 ≥75 요주의 / 본체 ≥95 이상 (국토안전관리원 온도패턴법)'),
 '계기용변성기(MOF)': (75, 95, 'PT·CT·MOF 단자 ≥75 / 본체 ≥95 (국토안전관리원 온도패턴법 준용)'),
 '단로기(DS)': (65, 80, '단로기 접촉부 ≥65 / 접속부 ≥80 (국토안전관리원 온도패턴법)'),
 '차단기': (60, 65, '차단기 단자 ≥60 / 본체 ≥65 (국토안전관리원 온도패턴법, 저압 MCCB 기준 준용)'),
 '전자접촉기': (65, 80, '접촉부(동) ≥65 요주의 (KESCO, KS C 4502); 접속부 ≥80 이상 (국토안전관리원)'),
 '변압기 접속부': (80, 95, '변압기 표면 80~95 요주의 / ≥95 이상 (국토안전관리원 온도패턴법)'),
 '변압기 케이블 연결개소 포함': (80, 95, '변압기 표면 80~95 요주의 / ≥95 이상; CV 케이블 ≥90 (국토안전관리원)'),
 '케이블&부스 연결개소': (65, 90, '부스바 접촉부 ≥65 / CV 케이블 ≥90 (국토안전관리원 온도패턴법)'),
 '접지용 콘덴서': (75, None, '콘덴서 단자 ≥75 (국토안전관리원 온도패턴법)'),
 '고압전동기': (None, 100, '베어링·씰·윤활유 한계 100°C (잠정, 제조사 확인 필요)'),
 '기어박스': (None, 100, '베어링·씰·윤활유 한계 100°C (잠정, 제조사 확인 필요)'),
 '펌프': (None, 100, '베어링·씰·윤활유 한계 100°C (잠정, 제조사 확인 필요)'),
 '차축베어링': (None, 100, '베어링·씰·윤활유 한계 100°C (잠정, EN 15437-1 핫박스 임계 미확인)'),
}
GATE_LOW_RELIABILITY = {'펌프': 0.285, '계기용변압기(PT)': 0.615, '단로기(DS)': 0.676}   # within-test RF AUROC < 0.7 (results/modellog.txt)
MISSING_COMMON = ['주변 기온 실측(카메라 설정값만 존재)', '운전 이력·이전 촬영', '방사율 실측']
MISSING_ELEC = MISSING_COMMON + ['부하율·전류', '상(R/S/T) 구분']
MISSING_BULK = MISSING_COMMON + ['운전 부하·회전수', '진동·윤활 이력']
MISSING_KO = MISSING_ELEC
def missing_for(f): return MISSING_BULK if f['equipment'] in BULK else MISSING_ELEC

def pattern_code(t):
    if t['hotspot_area_frac'] > 0.3: return '면형'
    if t['n_hotspots'] >= 3 and t['hotspot_area_frac'] < 0.3: return '다점형'
    if t['hotspot_area_frac'] < 0.08: return '점형'
    return '불명'

def sibling_contaminated(f):
    t = f['thermal']
    if not t['sibling_t_max']: return False
    over = sum(1 for s in t['sibling_t_max'] if s - t['scene_median'] > 5.5)
    return over >= 0.5 * len(t['sibling_t_max'])

def abs_limit(f):
    lim = ABS_LIMITS.get(f['equipment']); tm = f['thermal']['t_max']
    if not lim: return {'limit_warn': None, 'limit_abn': None, 'basis': '한계 미등록(확인 불가)', 'exceeded': None}
    w, a, b = lim
    ex = 'abnormal' if (a is not None and tm >= a) else ('warn' if (w is not None and tm >= w) else None)
    return {'limit_warn': w, 'limit_abn': a, 'basis': b, 'exceeded': ex}

def severity(f):
    """S/A/P axes + combination rules (deterministic)."""
    t = f['thermal']; g = f['gate']; n = f.get('norms', {}); eq = f['equipment']
    contaminated = sibling_contaminated(f); out = {'sibling_contaminated': contaminated, 'gate_grade_conflict': False}
    scene_lvl = level_scene(t['dT_scene_max']); out['scene_level'] = scene_lvl
    if eq in BULK and n.get('equipment_seen_in_training'):
        pct = n['t_max_pct_vs_normal']; lvl = level_population(pct, n.get('dT_scene_pct_vs_normal'))
        out.update(reference='population', reference_ko='동종 설비 정상 분포(절대온도) 대비(잠정 기준)', dT_ref=None, level_raw=lvl,
                   metric_ko=f"대상 최고온 {t['t_max']}°C = 정상 프레임 분포 백분위 {pct} (정상 중앙값 {n['normal_t_max_p50']}°C, P95 {n['normal_t_max_p95']}°C); 장면 대비 ΔT {t['dT_scene_max']}°C",
                   source=P1.SRC_DATA + "; ISO 18434-1 기준선 권고")
    elif eq in ELEC_CONN and t['dT_sibling_max'] is not None and t['n_siblings'] >= 1 and not contaminated:
        lvl = level_sibling(t['dT_sibling_max'])
        out.update(reference='sibling', reference_ko='같은 화면 동종 부품(형제) 최고온 중앙값 대비', dT_ref=t['dT_sibling_max'], level_raw=lvl,
                   metric_ko=f"형제 대비 ΔT {t['dT_sibling_max']}°C (형제 {t['n_siblings']}개, 기준 {t['sibling_reference']}°C); 장면 대비 ΔT {t['dT_scene_max']}°C",
                   source=P1.SRC_KR + "; " + P1.SRC_NETA_SIM)
    else:
        out.update(reference='scene', reference_ko='장면 중앙값(주변 온도 대용, 비표준 기준점) 대비' + (' — 형제 동반 과열로 형제 기준 무효' if contaminated else ''),
                   dT_ref=t['dT_scene_max'], level_raw=scene_lvl,
                   metric_ko=f"장면 대비 ΔT {t['dT_scene_max']}°C (장면 중앙값 {t['scene_median']}°C)", source=P1.SRC_NETA_AMB + " 준용")
    al = abs_limit(f); out['abs_limit'] = al
    flagged = g['rf_verdict'] == 'flag'
    if not flagged and al['exceeded'] == 'abnormal':
        out['level'] = 2; out['level_code'], out['level_ko'], out['action_ko'] = LEVELS[2]
        out['gate_grade_conflict'] = True
        out['metric_ko'] += '; 절대온도 한계 초과(' + str(t['t_max']) + '°C) — 게이트는 통과했으나 한계 초과로 검토 필요'
        return out
    if flagged:
        base = out['level_raw'] if out['level_raw'] is not None else scene_lvl
        if out['reference'] == 'population': base = max(base, scene_lvl - 1)
        if base == 0: out['gate_grade_conflict'] = True
        lvl_final = max(1, base)
        if al['exceeded'] == 'abnormal': lvl_final = max(lvl_final, 4 if out['reference'] == 'population' else 3)
    else:
        lvl_final = 0
    out['level'] = lvl_final; out['level_code'], out['level_ko'], out['action_ko'] = LEVELS[lvl_final]
    return out

def trigger_zone(f, sev=None):
    p = f['gate']['rf_prob']
    if p >= 0.5: return 'flag'
    if sev is not None and sev['abs_limit']['exceeded'] == 'abnormal': return 'review'
    if p >= 0.35 and f['gate']['rule_verdict'] == 'flag': return 'review'
    return 'pass'

def gate_reasons_v2(f, sev):
    """evidence list + separate caveat list. Equipment-relative wording; never presents a caveat as support."""
    t = f['thermal']; n = f.get('norms', {}); flagged = trigger_zone(f, sev) == 'flag'
    ev = []; cav = []
    dp = n.get('dT_scene_pct_vs_normal'); tp = n.get('t_max_pct_vs_normal')
    if dp is not None:
        if dp >= 95: ev.append('장면 대비 ΔT가 동종 설비 정상 분포 상위 5% 이내')
        elif dp >= 90: ev.append('장면 대비 ΔT가 동종 설비 정상 분포 상위 10% 이내')
        elif t['dT_scene_max'] > 5.5 and not flagged:
            cav.append('장면 대비 ΔT ' + str(t['dT_scene_max']) + '°C는 절대값으로는 크지만 동종 설비 정상 분포 내(백분위 ' + str(dp) + ')')
        elif t['dT_scene_max'] > 5.5:
            ev.append('장면 대비 ΔT ' + str(t['dT_scene_max']) + '°C (동종 설비 정상 분포 백분위 ' + str(dp) + ')')
    elif t['dT_scene_max'] > 5.5: ev.append('장면 대비 ΔT가 큼(설비별 정상 분포 기준 없음)')
    if tp is not None and tp >= 95: ev.append('절대온도가 동종 설비 정상 분포 상위 ' + str(round(100-tp, 1)) + '% 이내')
    if t['dT_sibling_max'] is not None and t['dT_sibling_max'] > 3:
        if sev.get('sibling_contaminated'): cav.append('형제 부품도 동반 과열이라 형제 기준은 무효(등급에 사용하지 않음)')
        else: ev.append('형제 부품 대비 ΔT ' + str(t['dT_sibling_max']) + '°C')
    if t['hotspot_area_frac'] < 0.08 and t['n_hotspots'] >= 1:
        if flagged: ev.append('열점이 국부에 집중')
        else: cav.append('열점은 국부에 집중되어 있으나 온도차는 정상 범위')
    if sev['abs_limit']['exceeded']:
        ev.append('절대온도 한계 ' + ('초과(이상)' if sev['abs_limit']['exceeded'] == 'abnormal' else '초과(요주의)'))
    if tp is not None and tp < 90 and not flagged: ev.append('동종 설비 정상 분포 범위 내')
    if not t['frame_max_inside_target'] and t['dT_scene_max'] > 5.5:
        cav.append('화면 최고온(' + str(t['scene_max']) + '°C)이 대상 밖에 있어 상자가 발열부를 온전히 포함하는지 확인 필요')
    bd = f.get('boundary') or {}
    if bd.get('external_source_suspected'):
        cav.append('최고온 지점이 상자 테두리에 있고 바로 바깥(' + str(bd['t_max_just_outside']) + '°C)이 더 뜨거움 — 인접 외부 열원이 상자 안으로 번진 값일 수 있음. 테두리 제외 시 최고온 ' + str(bd['t_max_eroded']) + '°C(장면 대비 ' + str(bd['dT_scene_eroded']) + '°C)로 ' + str(bd['dT_drop_when_border_removed']) + '°C 하락')
    elif bd.get('dT_drop_when_border_removed', 0) > 2:
        cav.append('테두리 ' + str(bd.get('inset_px', 3)) + 'px 제외 시 최고온이 ' + str(bd['dT_drop_when_border_removed']) + '°C 하락 — 경계 화소의 영향 확인 필요')
    if not ev: ev.append('특징 조합(형상·온도)에 의한 게이트 판정')
    return ev, cav

def case_type(f, sev):
    t = f['thermal']; s = f.get('session_agg', {}); zone = trigger_zone(f, sev)
    ff = s.get('flag_frac'); nm = s.get('n_frames_component_matched', 0)
    if zone == 'flag':
        n = f.get('norms', {}); tp = n.get('t_max_pct_vs_normal'); dp = n.get('dT_scene_pct_vs_normal')
        below_norm = (tp is not None and tp < 50) and (dp is not None and dp < 50)
        if (f.get('boundary') or {}).get('external_source_suspected'):
            return 'boundary_contaminated', '최고온이 상자 경계의 외부 열원일 가능성 — 재촬영 후 재판정 필요'
        if nm >= 3 and ff is not None and ff <= 0.34: return 'intermittent', '간헐적 플래그(세션 내 소수 프레임만 이상)'
        if sev['gate_grade_conflict']: return 'gate_only_shape', '온도차는 작으나 열 형상 근거로 플래그'
        if below_norm: return 'gate_flag_below_norm', '게이트는 플래그했으나 온도·온도차 모두 동종 설비 정상 중앙값 미만'
        if sev['level'] >= 2: return 'clear_overheat', '기준 초과 과열 소견'
        return 'mild_overheat', '경미한 기준 초과 소견'
    if zone == 'review': return 'review', '검토 필요(게이트 경계, 규칙은 이상)'
    if t['dT_scene_max'] > 5.5: return 'hot_but_normal', '장면 대비 고온이나 게이트 판정은 정상(동종 설비 정상 분포 내로 판단; 미탐 가능성 있음)'
    if t['dT_scene_max'] < 4: return 'no_thermal_evidence', '단일 프레임에 열적 이상 근거 없음'
    return 'normal', '정상 범위'

def subtle_flagged(f, sev):
    t = f['thermal']
    if trigger_zone(f, sev) != 'flag': return False
    prim = sev['dT_ref'] if sev['dT_ref'] is not None else t['dT_scene_max']
    shape = t['hotspot_area_frac'] < 0.08 and t['n_hotspots'] >= 1
    return prim < 4 and (t['dT_sibling_max'] is None or t['dT_sibling_max'] < 3) and not shape

def quality_flags(f, sev):
    q = []; b = f['bbox']; t = f['thermal']; c = f.get('calib') or {}
    if min(b['width'], b['height']) < 32: q.append('roi_small')
    if t['dT_sibling_max'] is None: q.append('sibling_absent')
    if sev['sibling_contaminated']: q.append('sibling_contaminated')
    if c and (c.get('emissivity') == 0.95 and c.get('obj_distance') == 1.0 and c.get('atmospheric_temp') == c.get('reflected_temp')): q.append('calib_default')
    q.append('rgb_unaligned')
    if not f.get('norms', {}).get('equipment_seen_in_training'): q.append('unseen_equipment')
    if f['equipment'] in GATE_LOW_RELIABILITY: q.append('gate_low_reliability_equipment')
    if sev['gate_grade_conflict']: q.append('gate_grade_conflict')
    if sev['abs_limit']['exceeded']: q.append('abs_limit_' + sev['abs_limit']['exceeded'])
    if t['hotspot_area_frac'] < 0.003 and t['peak_z'] > 4: q.append('reflection_suspect')
    bd = f.get('boundary') or {}
    if bd.get('external_source_suspected'): q.append('boundary_contamination_suspect')
    elif bd.get('peak_border_distance_px') is not None and bd['peak_border_distance_px'] < bd.get('inset_px', 3): q.append('peak_on_box_border')
    if subtle_flagged(f, sev): q.append('subtle_flagged')
    return q

def confidence(f, sev, qf):
    lvl, reason = P1.confidence(f, sev)
    cap = None
    if 'gate_low_reliability_equipment' in qf: cap = 'medium'; reason += f"; 설비군 게이트 신뢰도 낮음(RF AUROC {GATE_LOW_RELIABILITY[f['equipment']]})"
    if 'unseen_equipment' in qf: cap = 'medium'; reason += "; 미학습 설비"
    if 'subtle_flagged' in qf or 'reflection_suspect' in qf: cap = 'low'; reason += "; 단일 영상 근거 부족/반사 의심"
    if 'boundary_contamination_suspect' in qf: cap = 'low'; reason += "; 최고온이 상자 경계의 외부 열원일 가능성"
    if cap == 'medium' and lvl == 'high': lvl = 'medium'
    if cap == 'low': lvl = 'low'
    if cap == 'low' and ('여유 충분' in reason or '일관' in reason):
        # 2026-11-04: the parenthetical used to say 'gate probability itself is high' regardless of the verdict side; the
        # base reason's '여유 충분' also covers a confident PASS (probability far below the threshold). Judges flagged the
        # resulting self-contradiction on passed cases; it was present identically in both arms of every judged pair.
        side = '게이트 확률 자체는 높으나' if f['gate']['rf_prob'] >= 0.5 else '게이트 판정 자체는 여유 있으나'
        reason = reason.split(';')[-1].strip() + f' ({side} 위 사유로 신뢰도를 낮춤)'
    return lvl, reason

# ---------- band-word (number-free) facts view for pass-2 ----------
def band_word_dT(d, axis):
    if d is None: return '비교 불가(형제 부품 없음)'
    if axis == 'sib':
        if d <= 3: return '작음(정상 구간)'
        if d <= 5: return '약간 큼(주의 구간)'
        if d < 10: return '큼(요주의 구간)'
        if d <= 15: return '매우 큼(이상 구간)'
        return '극단적으로 큼(긴급 구간)'
    if d <= 5: return '작음(정상 구간)'
    if d <= 10: return '약간 큼(주의 구간)'
    if d <= 20: return '큼(요주의 구간)'
    if d <= 40: return '매우 큼(이상 구간)'
    return '극단적으로 큼(긴급 구간)'
def band_word_pct(p):
    if p is None: return '기준 없음(미학습 설비)'
    if p < 90: return '정상 분포 범위 내'
    if p < 95: return '정상 분포 상위권(주의)'
    if p < 99: return '정상 분포 상단 밖(요주의)'
    return '정상 분포 극단(이상)'
def voltage_class(spec):
    s = (spec or '').upper().replace(' ', '')
    m = re.search(r'(\d+(?:\.\d+)?)\s*KV', s)
    if m: v = float(m.group(1)); return '특고압' if v > 7 else '고압'
    m = re.search(r'(\d+)\s*V', s)
    if m: v = float(m.group(1)); return '저압' if v <= 1000 else '고압'
    return '확인 불가'
def session_word(s):
    nm = s.get('n_frames_component_matched', 0)
    if nm < 3: return '단일 프레임 근거(세션 내 동일 부품 매칭 부족)'
    ff = s['flag_frac']
    return '일관(세션 대부분 프레임에서 동일 판정)' if (ff >= 0.67 or ff <= 0.33) else '비일관(세션 내 판정 혼재)'

def _strip_digits(x):
    x = re.sub(r"\s*\(?[^()]*\d[^()]*\)?", "", x)
    return re.sub(r"\s+", " ", x).strip(" ;,·") or "판정 근거"

def facts_vlm(f, sev, ct, conf, obs):
    t = f['thermal']; n = f.get('norms', {}); s = f.get('session_agg', {}); zone = trigger_zone(f)
    verdict = {'flag': '이상 의심(플래그)', 'review': '검토 필요(경계)', 'pass': '정상 범위(통과)'}[zone]
    lines = [
        f"설비 종류: {f['equipment']} / 전압 구분: {voltage_class(f['spec'])} / 현장: {f['site']}",
        f"시스템 판정(변경 불가): {verdict}; 심각도: {sev['level_ko']}; 케이스 유형: {ct[1]}; 신뢰도: {conf[0]}",
        f"판정 근거: {', '.join(_strip_digits(x) for x in gate_reasons_v2(f, sev)[0])}",
        f"동종 부품 대비 온도차: {band_word_dT(t['dT_sibling_max'], 'sib')}" + ('; 형제 부품도 동반 과열 의심' if sev['sibling_contaminated'] else ''),
        f"장면(주변) 대비 온도차: {band_word_dT(t['dT_scene_max'], 'scene')}",
        f"동종 설비 정상 분포 대비: {band_word_pct(n.get('t_max_pct_vs_normal'))}",
        f"절대온도 한계: {'초과(이상)' if sev['abs_limit']['exceeded']=='abnormal' else ('초과(요주의)' if sev['abs_limit']['exceeded']=='warn' else ('미초과' if sev['abs_limit']['limit_abn'] or sev['abs_limit']['limit_warn'] else '확인 불가'))}",
        f"열 형태(계측): {pattern_code(t)}; 열점 위치(계측): {grid_ko(t['hotspot_position_in_target'])}",
        f"세션 증거: {session_word(s)}",
        f"관찰(일차 패스, 이미지 판독): 부위 {obs.get('part','확인 불가')}; 열점 위치 판독 {obs.get('p_grid','확인 불가')}; 열 형태 판독 {obs.get('heat_pattern','불명')}; 육안 특이 {', '.join(obs.get('visual_flags',['특이 없음']))}",
        f"확인 불가 항목(데이터 없음): {', '.join(missing_for(f))}",
    ]
    txt = "\n".join(lines)
    assert not re.search(r'\d', txt), f"digit leaked into FACTS_VLM: {txt}"
    return txt

# ---------- validation ----------
NUMWORD = re.compile(r'(?:[영일이삼사오육칠팔구십백천]+|하나|둘|셋|넷|다섯|여섯|일곱|여덟|아홉|열)\s*(?:도|퍼센트|배|번째|번|개|프레임|℃)')
def _bad_text(s, allow_numword=False):
    if re.search(r'\d', s): return 'digits'
    if CJK.search(s): return 'cjk'
    if re.search(r'[A-Za-z]{4,}', s): return 'latin'
    if not allow_numword and NUMWORD.search(s): return 'numword'
    if re.search(r'(번 단자|R상|S상|T상|[RST]-?상)', s): return 'phase_number'
    return None

def _frags(a):
    return [x.strip() for x in re.split(r'[·()/]', a) if len(x.strip()) >= 2]
def _enum_pick(v, allowed, fallback):
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    for a in allowed:                      # exact / full-key first
        key = a.split('(')[0]
        if key and key in s: return a, False
    for a in allowed:                      # then any fragment
        if any(fr in s for fr in _frags(a)): return a, False
    return fallback, True

def validate_pass1(vlm, f):
    eq = f['equipment']; parts = PARTS.get(eq, []) + ['확인 불가']; out = {}; viol = []
    vlm = vlm if isinstance(vlm, dict) else {}
    ep = str(vlm.get('equipment_plausible', '')); out['equipment_plausible'] = '예' if '예' in ep else ('아니오' if '아니' in ep else '확인 불가')
    if not vlm: viol.append('pass1:missing')
    out['part'], oov = _enum_pick(vlm.get('part', ''), parts, '확인 불가'); viol += ['part:oov'] if oov and vlm.get('part') else []
    pg = str(vlm.get('p_grid', '')); pick = None
    for gk in sorted(GRID_KO, key=len, reverse=True) + ['상자 밖']:   # longest first: '중앙' is a substring of '하단 중앙'
        if gk in pg: pick = gk; break
    if pick is None:
        # tolerate synonyms
        v = '상단' if ('상' in pg and '하' not in pg) else ('하단' if '하' in pg else '중단')
        h = '좌측' if '좌' in pg else ('우측' if '우' in pg else '중앙')
        pick = '중앙' if (v == '중단' and h == '중앙') else f"{v} {h}" if pg else '확인 불가'
        if pg and pick != '확인 불가': viol.append('p_grid:normalized')
    out['p_grid'] = pick
    hp, oov = _enum_pick(vlm.get('heat_pattern', ''), PATTERNS, '불명'); out['heat_pattern'] = hp; viol += ['heat_pattern:oov'] if oov and vlm.get('heat_pattern') else []
    vf = vlm.get('visual_flags', []); vf = vf if isinstance(vf, list) else [str(vf)]
    fl = [a for a in VISUAL_FLAGS if any(a.split('·')[0][:2] in str(x) for x in vf)]
    if not f['images'].get('rgb_crop'): fl = [x for x in fl if x != '가시광 없음'] + ['가시광 없음']
    # surface-condition claims require an ALIGNED visible image; ours is not aligned, so they cannot be grounded
    surface = {'표면 변색', '이완·부식 흔적', '케이블 손상', '이물질·오염'}
    if set(fl) & surface:
        viol.append('visual_flags:rgb_unaligned_sealed'); fl = [x for x in fl if x not in surface]
    out['visual_flags'] = fl or ['가시광 정렬 미검증으로 표면 소견 판독 불가']
    q, _ = _enum_pick(vlm.get('image_quality', ''), QUALITY, '보통'); out['image_quality'] = q
    pl = str(vlm.get('part_location_ko', '') or '')
    bad = _bad_text(pl) if pl else 'missing'
    if bad: viol.append('part_location_ko:' + bad); pl = ''
    out['part_location_ko'] = pl[:120]
    return out, viol

def cross_check(obs, f):
    t = f['thermal']; truth = grid_ko(t['hotspot_position_in_target']); pg = obs['p_grid']
    if pg in GRID_XY and truth in GRID_XY:
        (x1, y1), (x2, y2) = GRID_XY[pg], GRID_XY[truth]; d = max(abs(x1-x2), abs(y1-y2))
        sp = 'exact' if d == 0 else ('adjacent' if d == 1 else 'mismatch')
    else: sp = 'unknown'
    pc = pattern_code(t); hp = obs['heat_pattern']
    pat = 'match' if hp == pc else ('unknown' if (hp == '불명' or pc == '불명') else 'differ')
    return {'spatial_check': sp, 'p_grid_truth': truth, 'pattern_check': pat, 'pattern_code': pc}

POLAR_NEG = re.compile(r'(정상 범위|이상 없|양호|경미|적절|문제 없|이상이 없)')
POLAR_POS = re.compile(r'(과열|위험|소손|즉시|긴급|심각)')
def validate_pass2(vlm, f, sev, ct, obs, cc):
    vlm = vlm if isinstance(vlm, dict) else {}; out = {}; viol = []
    causes = CAUSES_BULK if f['equipment'] in BULK else CAUSES_ELEC
    if not vlm: viol.append('pass2:missing')
    ch = vlm.get('cause_hypothesis', []); ch = ch if isinstance(ch, list) else [str(ch)]
    picked = []
    for x in ch:
        c, oov = _enum_pick(str(x), causes, None)
        if c is None: viol.append('cause:oov'); continue
        if c not in picked: picked.append(c)
    pc = cc['pattern_code']; compat = COMPAT.get(pc)
    if compat is not None:
        bad = [c for c in picked if c not in compat]
        if bad: viol.append('cause:incompatible_with_pattern'); picked = [c for c in picked if c in compat]
    if 'subtle_flagged' in f.get('_qf', []): picked = ['확인 불가']
    if 'boundary_contamination_suspect' in f.get('_qf', []):
        viol.append('cause:suppressed_boundary_contamination'); picked = ['확인 불가']
    out['cause_hypothesis'] = (picked[:2] or ['확인 불가'])
    cb = vlm.get('cause_basis', []); cb = cb if isinstance(cb, list) else [str(cb)]
    basis = [b for b in BASIS if any(b[:4] in str(x) for x in cb)]
    t = f['thermal']; s = f.get('session_agg', {})
    if '동종 부품 대비' in basis and t['dT_sibling_max'] is None: viol.append('basis:sibling_absent'); basis.remove('동종 부품 대비')
    if '세션 증거' in basis and s.get('n_frames_component_matched', 0) < 3: viol.append('basis:session_absent'); basis.remove('세션 증거')
    _same = s.get('hotspot_pos_same_as_this_frame')
    if '세션 증거' in basis and _same is not None and _same < 0.5:
        viol.append('basis:session_position_inconsistent'); basis.remove('세션 증거')
    if '정상 분포 대비' in basis and not f.get('norms', {}).get('equipment_seen_in_training'): viol.append('basis:norms_absent'); basis.remove('정상 분포 대비')
    if '육안 특이' in basis and obs.get('visual_flags', ['특이 없음']) == ['특이 없음']: viol.append('basis:no_visual'); basis.remove('육안 특이')
    out['cause_basis'] = basis
    fc = vlm.get('field_checks', []); fc = fc if isinstance(fc, list) else [str(fc)]
    allowed_checks = CHECKS_BULK if f['equipment'] in BULK else CHECKS_ELEC
    checks = [a for a in allowed_checks if any(a.split('(')[0][:5] in str(x) for x in fc)]
    dropped = [str(x) for x in fc if not any(a.split('(')[0][:5] in str(x) for a in allowed_checks)]
    if dropped: viol.append('checks:out_of_group')
    if 'reflection_suspect' in f.get('_qf', []) or '반사 의심' in obs.get('visual_flags', []):
        if '각도 바꿔 재촬영(반사 확인)' not in checks: checks.insert(0, '각도 바꿔 재촬영(반사 확인)')
    out['field_checks'] = checks[:3]
    note = str(vlm.get('operator_note_ko', '') or ''); bad = _bad_text(note) if note else 'missing'
    zone = trigger_zone(f)
    if not bad:
        if zone == 'flag' and POLAR_NEG.search(note) and not re.search(r'(아니|않)', note): bad = 'polarity_under'
        if zone == 'pass' and POLAR_POS.search(note): bad = 'polarity_over'
        if any(m.split('(')[0][:3] in note for m in missing_for(f)[:1]) and re.search(r'(높|낮|초과|과부하)', note): bad = 'missing_asserted'
        other = CHECKS_ELEC if f['equipment'] in BULK else CHECKS_BULK
        if any(c.split('(')[0][:5] in note for c in other if c not in allowed_checks): bad = 'out_of_group'
    if bad: viol.append('operator_note_ko:' + bad); note = template_note(f, ct)
    out['operator_note_ko'] = note[:120]
    return out, viol

NOTE_ELEC = {'clear_overheat': '현장에서 해당 부위의 체결 상태와 부하 전류를 확인하십시오.', 'mild_overheat': '다음 점검 시 동일 부위를 재촬영하여 추세를 확인하십시오.',
             'intermittent': '세션 내 판정이 비일관하므로 부하 조건을 기록하며 재촬영하십시오.', 'gate_only_shape': '온도차는 작으나 열 형상 근거로 플래그되었으므로 접촉면·체결 상태를 확인하십시오.',
             'review': '게이트 경계 케이스이므로 부하 조건을 기록하고 재촬영 후 판단하십시오.', 'hot_but_normal': '설비 특성상 고온이 정상 범위일 수 있으므로 절대온도 추세만 기록하십시오.',
             'no_thermal_evidence': '단일 프레임에서 열적 근거가 없어 추가 판단은 유보합니다.', 'normal': '정상 감시를 유지하십시오.'}
NOTE_BULK = {'clear_overheat': '현장에서 해당 부위의 진동·윤활 상태와 운전 부하를 확인하십시오.', 'mild_overheat': '다음 점검 시 동일 부위를 재촬영하여 온도 추세를 확인하십시오.',
             'intermittent': '세션 내 판정이 비일관하므로 운전 조건을 기록하며 재촬영하십시오.', 'gate_only_shape': '온도차는 작으나 국부 열점 형상으로 플래그되었으므로 베어링·씰 부위를 점검하십시오.',
             'review': '게이트 경계 케이스이므로 운전 조건을 기록하고 재촬영 후 판단하십시오.', 'hot_but_normal': '상시 고온 설비이므로 절대온도 추세와 윤활 상태만 기록하십시오.',
             'no_thermal_evidence': '단일 프레임에서 열적 근거가 없어 추가 판단은 유보합니다.', 'normal': '정상 감시를 유지하십시오.'}
ACTION_BULK = {0: '정상 감시 유지', 1: '기록 후 다음 정기점검 시 재확인',
               2: '정밀점검 계획 수립(수 주 내), 진동·윤활 상태 확인',
               3: '조속한 정비(가능한 빠른 시기), 운전 부하 저감 검토',
               4: '즉시 조치(정지 검토 후 베어링·윤활 계통 정비)'}
def action_for(f, sev):
    return ACTION_BULK[sev['level']] if f['equipment'] in BULK else sev['action_ko']

NOTE_EXTRA = {'boundary_contaminated': '최고온 지점이 상자 경계에 있고 바로 바깥이 더 뜨겁습니다. 촬영 각도와 상자 범위를 조정해 재촬영한 뒤 판단하십시오.',
              'gate_flag_below_norm': '탐지 모델은 플래그했으나 온도·온도차 모두 동종 설비 정상 중앙값 미만입니다. 기록만 하고 다음 점검 시 재확인하십시오.'}
def template_note(f, ct):
    if ct[0] in NOTE_EXTRA: return NOTE_EXTRA[ct[0]]
    return (NOTE_BULK if f['equipment'] in BULK else NOTE_ELEC).get(ct[0], '정상 감시를 유지하십시오.')

def thermal_finding_ko(f, sev, cc):
    t = f['thermal']; pc = cc['pattern_code']; pos = grid_ko(t['hotspot_position_in_target'])
    s = f"대상 최고 온도 {t['t_max']}°C, 장면 중앙값 {t['scene_median']}°C(장면 대비 {t['dT_scene_max']}°C). 열점은 대상 내 {pos}에 {pc} 분포"
    if t['dT_sibling_max'] is not None: s += f"; 동종 부품 대비 {t['dT_sibling_max']}°C"
    if not t['frame_max_inside_target']: s += f"; 화면 최고온({t['scene_max']}°C)은 대상 밖"
    return s + "."

def reconcile(obs, cc, viol):
    """code measurement wins over the VLM reading; disagreements are recorded, never printed side by side."""
    o = dict(obs)
    if cc['pattern_check'] == 'differ':
        o['heat_pattern'] = cc['pattern_code']; viol.append('heat_pattern:overridden_by_measurement')
    if cc['spatial_check'] != 'exact' and o.get('p_grid') != cc['p_grid_truth']:
        o['p_grid'] = cc['p_grid_truth']; o['part_location_ko'] = ''
        viol.append('p_grid:overridden_by_measurement')
    return o

def _session_line(f):
    sa = f.get('session_agg') or {}
    if sa.get('n_frames_component_matched', 0) < 1: return '세션 내 동일 부품 매칭 없음(단일 프레임 근거)'
    line = f"같은 세션 {sa['n_frames_in_session']}프레임 중 동일 부품 매칭 {sa['n_frames_component_matched']}프레임, 그중 {sa.get('n_flagged', 0)}프레임 플래그"
    same = sa.get('hotspot_pos_same_as_this_frame')
    if same is not None and same < 0.5:
        line += f"; 다만 이 프레임의 열점 위치는 세션 내 {round(100*(1-same))}% 프레임과 다름(최빈 위치 {pos_ko(sa.get('hotspot_pos_mode', ''))}) — 고정 결함이 아닐 수 있음"
    return line

def suppress_unsupported(f, p2, viol):
    """causes/bases that the measurement cannot support, regardless of which arm produced them."""
    qf = f.get('_qf', []); sa = f.get('session_agg') or {}
    if 'boundary_contamination_suspect' in qf or 'subtle_flagged' in qf:
        if p2.get('cause_hypothesis') != ['확인 불가']:
            viol.append('cause:suppressed_unsupported_measurement')
        p2['cause_hypothesis'] = ['확인 불가']; p2['cause_basis'] = []
    same = sa.get('hotspot_pos_same_as_this_frame')
    if same is not None and same < 0.5 and '세션 증거' in p2.get('cause_basis', []):
        viol.append('basis:session_position_inconsistent'); p2['cause_basis'] = [b for b in p2['cause_basis'] if b != '세션 증거']
    if p2.get('cause_hypothesis') == ['확인 불가']: p2['cause_basis'] = []
    if '육안 특이' in p2.get('cause_basis', []):
        vf = (p2.get('_visual_flags') or [])
        if not any(v in ('반사 의심',) for v in vf):
            viol.append('basis:no_visual'); p2['cause_basis'] = [b for b in p2['cause_basis'] if b != '육안 특이']
    return p2

def assemble(f, sev, ct, conf, qf, obs, cc, p2, viol, raw):
    t = f['thermal']; g = f['gate']; zone = trigger_zone(f, sev)
    verdict = {'flag': '이상 의심', 'review': '검토 필요', 'pass': '정상 범위'}[zone]
    al = sev['abs_limit']; obs = reconcile(obs, cc, viol)
    p2 = dict(p2); p2['_visual_flags'] = obs.get('visual_flags', []); p2 = suppress_unsupported(f, p2, viol); p2.pop('_visual_flags', None)
    ev, cav = gate_reasons_v2(f, sev)
    return {
        'case_id': f['case_id'], 'equipment': f['equipment'], 'site': f['site'], 'spec': f['spec'],
        'verdict': verdict, 'severity': f"{sev['level_code']} {sev['level_ko']}", 'severity_level': sev['level'],
        'severity_reference': sev['reference_ko'], 'severity_metric': sev['metric_ko'], 'severity_source': sev['source'],
        'scene_level': sev['scene_level'], 'gate_grade_conflict': sev['gate_grade_conflict'],
        'abs_limit_line': (f"절대온도 한계 {al['exceeded']=='abnormal' and '초과(이상)' or '초과(요주의)'}: 최고온 {t['t_max']}°C — {al['basis']}" if al['exceeded'] else (f"절대온도 한계 미초과 — {al['basis']}" if (al['limit_abn'] or al['limit_warn']) else '절대온도 한계: 확인 불가')),
        'measurements': {'t_max': t['t_max'], 't_mean': t['t_mean'], 'scene_median': t['scene_median'], 'dT_scene': t['dT_scene_max'],
                         'sibling_reference': t['sibling_reference'], 'dT_sibling': t['dT_sibling_max'], 'n_siblings': t['n_siblings'],
                         'hotspot_area_frac': t['hotspot_area_frac'], 'rf_prob': g['rf_prob'],
                         't_max_eroded': (f.get('boundary') or {}).get('t_max_eroded'),
                         'dT_scene_eroded': (f.get('boundary') or {}).get('dT_scene_eroded')},
        'thermal_finding_ko': thermal_finding_ko(f, sev, cc), 'gate_reasons_ko': ev, 'caveats_ko': cav,
        'case_type': ct[0], 'case_type_ko': ct[1], 'confidence': conf[0], 'confidence_reason': conf[1],
        'action_ko': (('상자 경계 외부 열원 오염 의심 — 촬영 각도·상자 범위를 조정해 재촬영 후 재판정(현 등급은 잠정)'
                       if (f.get('boundary') or {}).get('external_source_suspected') else action_for(f, sev)) if zone == 'flag' else
                      ('부하 조건 기록 후 재촬영하여 재판정(검토 큐)' if zone == 'review' else
                       (('절대온도 추세 기록(상시 고온 설비)' if f['equipment'] in BULK else '절대온도 추세 기록 후 다음 점검 시 재확인') if ct[0] == 'hot_but_normal' else '정상 감시 유지')))
                     + ('; 신뢰도 낮음 — 재촬영 권고' if conf[0] == 'low' and zone == 'pass' else ''),
        'field_checks_ko': p2.get('field_checks', []),
        'session_evidence_line': _session_line(f),
        'limitations_ko': ['단일 촬영 기준', '온도차 4°C 미만의 이상은 놓칠 수 있음'] + [f"{q}" for q in qf], 'missing_ko': missing_for(f),
        'spatial_check': cc['spatial_check'], 'pattern_check': cc['pattern_check'],
        'vlm_pass1': obs, 'vlm_pass2': p2, 'validation': {'violations': viol, 'fallback_fields': sorted(set(v.split(':')[0] for v in viol))},
    }


# ---------- A0+ enhanced code-only template (strong no-VLM baseline for the usefulness comparison) ----------
PART_BY_ROW = {   # equipment -> (top, middle, bottom) most likely part at the hotspot row
 '계기용변류기(CT)': ('일차 단자 접속부', '본체(절연 몰드)', '이차 단자함'), '계기용변압기(PT)': ('일차 단자 접속부', '본체(절연 몰드)', '이차 단자함'),
 '계기용변성기(MOF)': ('단자 접속부', '본체(탱크)', '이차 단자함'), '단로기(DS)': ('접촉부(블레이드·클립)', '단자 접속부', '애자'),
 '차단기': ('상부 단자 접속부', '소호부·본체', '하부 단자 접속부'), '전자접촉기': ('상부 단자', '코일·본체', '하부 단자'),
 '변압기 접속부': ('부싱 단자', '리드 접속부', '케이블 접속부'), '케이블&부스 연결개소': ('부스바 접속부', '볼트 체결부', '케이블 슈 접속부'),
 '변압기 케이블 연결개소 포함': ('부싱', '케이블 접속부', '탱크 표면'), '접지용 콘덴서': ('단자 접속부', '본체', '부싱'),
 '신재생 설비': ('단자·접속부', '본체·모듈', '케이블'), '고압전동기': ('단자함', '고정자 프레임', '구동측 베어링 하우징'),
 '기어박스': ('입력축 베어링부', '하우징 본체', '출력축 베어링부'), '펌프': ('커플링부', '펌프 케이싱', '베어링 하우징'),
 '보일러': ('배관 접속부', '노벽·케이싱', '점검구·도어'), '차축베어링': ('차축 베어링(액슬박스)', '차축 본체', '휠 림·답면'),
}
CAUSE_PRIOR = {('ELEC', '점형'): ['접촉부 접촉저항 증가(체결 이완·산화)', '접촉면 오염'], ('ELEC', '다점형'): ['접촉부 접촉저항 증가(체결 이완·산화)', '과부하 또는 상 불평형'],
               ('ELEC', '면형'): ['과부하 또는 상 불평형', '냉각·통풍 불량'], ('ELEC', '선형'): ['과부하 또는 상 불평형', '접촉부 접촉저항 증가(체결 이완·산화)'], ('ELEC', '불명'): ['확인 불가'],
               ('BULK', '점형'): ['베어링 손상·윤활 불량', '씰 마찰·누설'], ('BULK', '다점형'): ['베어링 손상·윤활 불량', '축 정렬 불량'],
               ('BULK', '면형'): ['과부하 운전', '냉각 불량'], ('BULK', '선형'): ['축 정렬 불량', '과부하 운전'], ('BULK', '불명'): ['확인 불가']}
CHECK_PRIOR = {'ELEC': {1: ['부하 조건 기록 후 재촬영'], 2: ['부하 전류·불평형률 측정', '체결 토크 점검'], 3: ['부하 전류·불평형률 측정', '체결 토크 점검', '접촉저항 측정'], 4: ['부하 전류·불평형률 측정', '접촉저항 측정', '접촉면 청소·산화 확인']},
               'BULK': {1: ['부하 조건 기록 후 재촬영'], 2: ['진동 측정', '윤활 상태·유량 확인'], 3: ['진동 측정', '윤활 상태·유량 확인', '축 정렬 확인'], 4: ['진동 측정', '윤활 상태·유량 확인', '냉각·통풍 상태 확인']}}

def a0plus_fields(f, sev, ct, cc):
    """code-only 'observations' and 'reasoning' built from CSV facts + lookup tables (no VLM)."""
    t = f['thermal']; grp = 'BULK' if f['equipment'] in BULK else 'ELEC'; pos = grid_ko(t['hotspot_position_in_target'])
    row = 0 if pos.startswith('상단') else (2 if pos.startswith('하단') else 1)
    part = PART_BY_ROW.get(f['equipment'], ('확인 불가',)*3)[row]
    pc = cc['pattern_code']; zone = trigger_zone(f)
    obs = {'equipment_plausible': '확인 불가', 'part': part, 'p_grid': pos, 'heat_pattern': pc, 'visual_flags': ['가시광 미판독'], 'image_quality': '확인 불가',
           'part_location_ko': f"최고온 지점은 대상 {pos}에 있으며 조회표상 {part} 부위로 추정됩니다."}
    causes = CAUSE_PRIOR.get((grp, pc), ['확인 불가']) if zone == 'flag' else ['정상 부하 발열' if grp == 'ELEC' else '정상 운전 발열']
    if 'subtle_flagged' in f.get('_qf', []): causes = ['확인 불가']
    checks = CHECK_PRIOR[grp].get(sev['level'], []) if zone == 'flag' else (['부하 조건 기록 후 재촬영'] if zone == 'review' else [])
    if 'reflection_suspect' in f.get('_qf', []): checks = ['각도 바꿔 재촬영(반사 확인)'] + checks
    p2 = {'cause_hypothesis': causes[:2], 'cause_basis': ['열 형태'] + (['동종 부품 대비'] if t['dT_sibling_max'] is not None and t['dT_sibling_max'] > 3 else []) + (['세션 증거'] if f.get('session_agg', {}).get('n_frames_component_matched', 0) >= 3 else []),
          'field_checks': checks[:3], 'operator_note_ko': template_note(f, ct)}
    return obs, p2
