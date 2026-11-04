# -*- coding: utf-8 -*-
"""render_report.py — assembled report JSON -> operator-facing Korean text (same layout for every arm).
usage: python render_report.py --tag v2 --cond v2 [--cases_dir DIR]  -> writes <out>/<cond>/<case>.txt and prints one sample."""
import json, os, sys, io, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from common import CASES
OUT = r"<PROJECT_ROOT>\results\vlm_out"

def fmt(v): return '확인 불가' if v is None else v

def render(rep, show_validation=True):
    m = rep['measurements']; v1 = rep['vlm_pass1']; v2 = rep['vlm_pass2']
    L = []
    L.append(f"■ 설비: {rep['equipment']} / 현장: {rep['site']} / 사양: {fmt(rep['spec'])}")
    L.append(f"■ 판정: {rep['verdict']}   심각도: {rep['severity']}   케이스 유형: {rep['case_type_ko']}")
    L.append(f"   등급 기준: {rep['severity_reference']} — {rep['severity_metric']}")
    L.append(f"   출처: {rep['severity_source']}")
    L.append(f"   {rep['abs_limit_line']}")
    L.append(f"■ 계측: 최고 {m['t_max']}°C / 평균 {m['t_mean']}°C / 장면 중앙값 {m['scene_median']}°C / 장면 대비 ΔT {m['dT_scene']}°C / 형제 대비 ΔT {fmt(m['dT_sibling'])} (형제 {m['n_siblings']}개) / 게이트 확률 {m['rf_prob']}")
    if m.get('t_max_eroded') is not None and m['t_max_eroded'] < m['t_max'] - 0.5:
        L.append(f"   테두리 제외 재계산: 최고 {m['t_max_eroded']}°C / 장면 대비 ΔT {m['dT_scene_eroded']}°C")
    L.append(f"■ 열 소견: {rep['thermal_finding_ko']}")
    L.append(f"   판정 근거: {', '.join(rep['gate_reasons_ko'])}")
    if rep.get('caveats_ko'): L.append(f"   유의 사항: {'; '.join(rep['caveats_ko'])}")
    L.append(f"   세션 증거: {rep['session_evidence_line']}")
    L.append(f"■ 이미지 판독: 부위 {v1['part']} / 열점 위치 판독 {v1['p_grid']} (계측 {rep.get('spatial_check','')}) / 열 형태 {v1['heat_pattern']} / 육안 {', '.join(v1['visual_flags'])} / 화질 {v1['image_quality']}")
    if v1.get('part_location_ko'): L.append(f"   {v1['part_location_ko']}")
    L.append(f"■ 원인 후보(추정): {', '.join(v2['cause_hypothesis'])}   근거: {', '.join(v2['cause_basis']) or '없음'}")
    L.append(f"■ 권고 조치: {rep['action_ko']}")
    if v2.get('field_checks'): L.append(f"   현장 확인 항목: {', '.join(v2['field_checks'])}")
    if v2.get('operator_note_ko'): L.append(f"   작업자 메모: {v2['operator_note_ko']}")
    L.append(f"■ 신뢰도: {rep['confidence']} — {rep['confidence_reason']}")
    L.append(f"■ 한계: {', '.join(rep['limitations_ko'])}; 미확인: {', '.join(rep['missing_ko'])}")
    if show_validation:
        v = rep['validation']; nv = len(v['violations'])
        if nv == 0: L.append("■ 기계 검증: 통과(모든 서술 항목이 계측과 정합)")
        else:
            ko = {'cause': '원인 후보', 'basis': '근거', 'checks': '확인 항목', 'operator_note_ko': '작업자 메모',
                  'heat_pattern': '열 형태', 'p_grid': '열점 위치', 'part_location_ko': '위치 서술', 'visual_flags': '육안 소견',
                  'pass1': '이미지 판독', 'pass2': '추론'}
            names = sorted({ko.get(x.split(':')[0], x.split(':')[0]) for x in v['violations']})
            L.append(f"■ 기계 검증: {nv}건 불합격 → {', '.join(names)} 항목은 이미지 판독 대신 계측 기반 기본값으로 대체됨")
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='v2'); ap.add_argument('--cond', default='v2'); ap.add_argument('--cases_dir', default=CASES); a = ap.parse_args()
    cd = os.path.join(OUT, a.tag, a.cond); n = 0; sample = None
    for fn in sorted(os.listdir(cd)):
        if not fn.endswith('.json'): continue
        r = json.load(open(os.path.join(cd, fn), encoding='utf-8')); txt = render(r['report'])
        open(os.path.join(cd, fn[:-5] + '.txt'), 'w', encoding='utf-8').write(txt); n += 1
        if sample is None: sample = txt
    print(f"rendered {n} reports in {cd}\n\n{sample}")

if __name__ == '__main__':
    main()
