# -*- coding: utf-8 -*-
"""build_noimg_tasks.py — NO-IMAGE (facts-only) frontier control arm.
For every locked case, build ONE text-only prompt that asks the model to fill the same descriptive fields the
image-based pipeline fills (pass-1 observation fields + pass-2 reasoning fields) from the band-worded, number-free
facts alone. No image is attached. If blinded judges prefer these reports over the code-only report as often as
they prefer the image-based frontier reports, the judged 'value' is fluency, not grounding.
Writes results/vlm_out/frontier/tasks_noimg.json (list of {case_id, prompt, out}); outputs go to
results/vlm_out/frontier/noimg/<case>.json with {"raw": "<model JSON text>"}.
"""
import json, os, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import protocol_v2 as P
L = r"<PROJECT_ROOT>\results\cases_lock"; F = r"<PROJECT_ROOT>\results\vlm_out\frontier"
OUT = os.path.join(F, 'noimg'); os.makedirs(OUT, exist_ok=True)
TEMPLATE_OBS = {'equipment_plausible': '확인 불가', 'part': '확인 불가', 'p_grid': '확인 불가', 'heat_pattern': '불명', 'visual_flags': ['가시광 없음'], 'image_quality': '보통', 'part_location_ko': ''}
PROMPT = """당신은 발전소 설비 열화상 진단 보조원입니다. 이번 과제에는 **이미지가 제공되지 않습니다.** 아래 [사실](시스템이 계측·확정한 내용, 수치 없음)만 근거로, 이미지 기반 보고서가 채우는 것과 같은 항목을 작성하십시오. 판정·심각도·수치는 시스템이 확정했으며 바꿀 수 없습니다.

절대 규칙: 숫자 금지, 한국어만, 판정을 바꾸지 말 것, [사실]에 "확인 불가"인 항목을 근거로 쓰지 말 것, 보지 않은 것을 본 것처럼 쓰지 말 것. 아래 JSON 하나만 출력.

등록된 설비 종류: {equipment}

[사실]
{facts}

작성 항목:
- part: 열점이 놓인 세부 부위로 가장 가능성 높은 것. 다음 중 하나 — {parts}
- heat_pattern: 열점 형태. 다음 중 하나 — 점형(작고 국부) / 면형(부품 전체가 고온) / 선형(도체·케이블 방향으로 길게) / 다점형(여러 개) / 불명
- part_location_ko: 열점 위치를 부품 기준으로 한 문장(숫자·번호·상 표기 없이)
- cause_hypothesis: 다음 후보 중 최대 두 개(가능성 순) — {causes}
- cause_basis: 각 원인을 지지하는 근거 항목만 — 동종 부품 대비 / 열 형태 / 육안 특이 / 세션 증거 / 정상 분포 대비
- field_checks: 현장 확인 항목 최대 세 개 — {checks}
- operator_note_ko: 현장 작업자에게 전할 한 문장(숫자 없이, 판정과 모순되지 않게)

{{
 "part": "...",
 "heat_pattern": "...",
 "part_location_ko": "...",
 "cause_hypothesis": ["..."],
 "cause_basis": ["..."],
 "field_checks": ["..."],
 "operator_note_ko": "..."
}}
"""
idx = [json.loads(l) for l in open(os.path.join(L, 'index.jsonl'), encoding='utf-8')]
tasks = []
for it in idx:
    cid = it['case_id']; f = json.load(open(os.path.join(L, cid, 'facts.json'), encoding='utf-8'))
    sev = P.severity(f); ct = P.case_type(f, sev); qf = P.quality_flags(f, sev); f['_qf'] = qf; conf = P.confidence(f, sev, qf)
    facts = P.facts_vlm(f, sev, ct, conf, TEMPLATE_OBS)
    facts = "\n".join(l for l in facts.split("\n") if not l.startswith('관찰('))   # no pass-1 observation exists in this arm
    assert not re.search(r'\d', facts)
    parts = ' / '.join(P.PARTS.get(f['equipment'], []) + ['확인 불가'])
    causes = ' / '.join(P.CAUSES_BULK if f['equipment'] in P.BULK else P.CAUSES_ELEC)
    tasks.append({'case_id': cid, 'cond': 'noimg', 'equipment': f['equipment'], 'images': [],
                  'prompt': PROMPT.format(equipment=f['equipment'], facts=facts, parts=parts, causes=causes, checks=' / '.join(P.CHECKS)),
                  'out': os.path.join(OUT, cid + '.json')})
json.dump(tasks, open(os.path.join(F, 'tasks_noimg.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f"wrote {len(tasks)} no-image tasks -> {os.path.join(F, 'tasks_noimg.json')}")
print(tasks[0]['prompt'][:900])
