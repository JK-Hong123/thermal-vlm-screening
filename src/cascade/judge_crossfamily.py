# -*- coding: utf-8 -*-
"""judge_crossfamily.py — re-run the frontier-VLM-vs-A0+ blind pairwise judging (the 32 locked
cases in results/judge_frontier/) with a judge model from a DIFFERENT vendor family than the
subject model (which was Claude), to test whether the "preferred 21/32" result is an artefact of
the judge and the frontier-VLM subject sharing a model family (self-preference bias).

Usage:
  # 1) Dry run (no API key needed): export a self-contained packet per case (images + both
  #    blinded reports + the exact instructions) for manual judging via any chat UI (ChatGPT,
  #    Gemini web, etc.) the user has access to.
  python judge_crossfamily.py --mode export

  # 2) Automatic, if an API key is available:
  set OPENAI_API_KEY=...   (or GOOGLE_API_KEY= / GEMINI_API_KEY=)
  python judge_crossfamily.py --mode run --provider openai --model gpt-4o
  python judge_crossfamily.py --mode run --provider gemini --model gemini-2.5-pro

  # 3) After manual judging (mode=export), paste each model's JSON answers back as
  #    results/judge_crossfamily_manual/caseXXX.json and then:
  python judge_crossfamily.py --mode aggregate --tag manual

Output is written to results/judge_crossfamily_<tag>/caseXXX.json (tag = provider or 'manual'),
and the aggregate summary to results/judge_crossfamily_<tag>_summary.txt, in a format directly
comparable to results/judge_frontier_summary.txt (same preference-rate / usefulness-gap metrics).
"""
import argparse, base64, json, os, sys, io, glob, collections, statistics as st, time
import numpy as np

R = r"<PROJECT_ROOT>\results"
LOCK = os.path.join(R, "cases_lock")
KEYF = os.path.join(R, "judge_frontier", "_key.json")
V2_DIR = os.path.join(R, "vlm_out", "v2frontier", "v2")
A0_DIR = os.path.join(R, "vlm_out", "v2frontier", "a0plus")

INSTRUCTIONS = """\
당신은 산업 열화상(적외선) 설비진단 보고서 두 건을 블라인드로 비교 평가하는 독립 심사위원입니다.
이 과제는 '어느 보고서 작성자가 어떤 AI 모델인지' 전혀 모른 채, 오직 아래 이미지와 두 보고서 텍스트만 보고 판단해야 합니다.

[배경]
- 이미지는 열화상 CCTV로 촬영한 설비(철도/발전소/변전소/소각장)의 크롭입니다. 흰 사각형 테두리(마커)가 진단 대상 부품의 위치를 표시합니다.
- 두 번째 이미지는 같은 장면의 가시광(RGB) 사진입니다. 열화상 카메라와 별도 카메라로 찍혀 화소 단위로 정렬되어 있지 않고, 종종 어둡거나 열화상과 무관한 각도일 수 있습니다.
- 두 보고서(A, B)는 같은 사례에 대해 서로 다른 방식으로 작성되었습니다. 판정·등급·측정수치는 두 보고서 모두 동일한 코드(규칙 기반 계산)가 생성했으므로 100% 동일합니다. 차이가 있다면 '이미지 판독' 부분과 일부 서술(소견/권고 문구)뿐입니다.

[보고서 A]
{report_a}

[보고서 B]
{report_b}

[요청 — 반드시 아래 JSON 형식으로만 답하세요. 다른 텍스트를 덧붙이지 마세요.]
{{
  "which_report_reads_image_more": "A" 또는 "B" 또는 "cannot_tell"   // 어느 보고서가 이미지를 실제로 더 많이/정확히 참조해 작성된 것처럼 보이는지 당신의 추정 (정답을 맞히라는 뜻이 아니라, 당신이 보기에 어느 쪽이 그렇게 '보이는지'),
  "preferred_report": "A" 또는 "B" 또는 "tie"   // 현장 기술자에게 더 유용한 보고서는 어느 쪽인가,
  "preference_reason": "한두 문장으로 그 이유",
  "usefulness_A_1to5": 1~5 정수,
  "usefulness_B_1to5": 1~5 정수,
  "actionability_A_1to5": 1~5 정수,
  "actionability_B_1to5": 1~5 정수,
  "adds_value_relative_to_other": "A가 B에 없는 유용한 정보를 더함" 또는 "B가 A에 없는 유용한 정보를 더함" 또는 "둘 다 동등" 또는 "둘 다 추가 정보 없음",
  "ungrounded_claims_A": 정수 (이미지나 제공된 수치로 뒷받침되지 않는, 혹은 이미지와 모순되는 서술의 개수),
  "ungrounded_claims_B": 정수,
  "trustworthy_for_operator": "A" 또는 "B" 또는 "both" 또는 "neither",
  "notes": "자유 기술(선택)"
}}
"""

# "Full" schema: faithfully reconstructs the original judge_frontier protocol's forced-grounding
# design. The judge must FIRST independently read the thermal image and state its own marker-cell
# and part reading (before being told anything about the two reports' content on that point), THEN
# compare. This lets ungrounded-claim counts be checked against an independent reading instead of
# being a pure vibe judgment -- this is exactly what the simplified schema (used 2026-10-15) dropped,
# which a later adversarial review flagged as a schema regression that makes the two runs non-comparable.
INSTRUCTIONS_FULL = """\
당신은 산업 열화상(적외선) 설비진단 보고서 두 건을 블라인드로 비교 평가하는 독립 심사위원입니다.
이 과제는 '어느 보고서 작성자가 어떤 AI 모델인지' 전혀 모른 채 진행합니다. 두 단계로 답하세요.

[배경]
- 이미지는 열화상 CCTV로 촬영한 설비(철도/발전소/변전소/소각장)의 크롭입니다. 흰 사각형 테두리(마커)가 진단 대상 부품의 위치를 표시합니다.
- 두 번째 이미지는 같은 장면의 가시광(RGB) 사진입니다. 열화상 카메라와 별도 카메라로 찍혀 화소 단위로 정렬되어 있지 않고, 종종 어둡거나 열화상과 무관한 각도일 수 있습니다.
- 두 보고서(A, B)는 같은 사례에 대해 서로 다른 방식으로 작성되었습니다. 판정·등급·측정수치는 두 보고서 모두 동일한 코드(규칙 기반 계산)가 생성했으므로 100% 동일합니다. 차이가 있다면 '이미지 판독' 부분과 일부 서술(소견/권고 문구)뿐입니다.

[1단계 — 보고서를 보기 전에, 당신 스스로 이미지를 먼저 판독하세요]
- 마커(흰 사각형)가 열화상 이미지의 3x3 격자(상단-좌/중/우, 중단-좌/중/우, 하단-좌/중/우) 중 어디에 있습니까?
- 마커가 가리키는 부위는 무엇으로 보입니까(가능하면 가시광도 참고)? 확신 정도는?

[2단계 — 아래 두 보고서를 읽고 비교하세요]
[보고서 A]
{report_a}

[보고서 B]
{report_b}

보고서 안에 적힌 '열점 위치'나 '부위' 서술이 1단계에서 당신이 직접 읽은 것과 맞는지 대조하세요.
보고서의 다른 서술(세션 반복성, 육안 소견 등)이 보고서 자체에 적힌 다른 수치나 유의사항과 모순되지 않는지도 확인하세요(자기모순이면 ungrounded로 셀 것).

[요청 — 반드시 아래 JSON 형식으로만 답하세요. 다른 텍스트를 덧붙이지 마세요.]
{{
  "my_marker_cell": "상단-좌" 등 3x3 격자 중 하나,
  "my_part": "당신이 추정한 부위(자유 서술)",
  "my_part_confidence": "high" 또는 "medium" 또는 "low",
  "which_report_reads_image_more": "A" 또는 "B" 또는 "cannot_tell",
  "preferred_report": "A" 또는 "B" 또는 "tie",
  "preference_reason": "한두 문장",
  "usefulness_A_1to5": 1~5 정수,
  "usefulness_B_1to5": 1~5 정수,
  "actionability_A_1to5": 1~5 정수,
  "actionability_B_1to5": 1~5 정수,
  "adds_value_relative_to_other": "A가 B에 없는 유용한 정보를 더함" 또는 "B가 A에 없는 유용한 정보를 더함" 또는 "둘 다 동등" 또는 "둘 다 추가 정보 없음",
  "ungrounded_claims_A": 정수 (1단계 당신의 독립 판독 또는 보고서 자체의 다른 서술과 모순되는 문장의 개수),
  "ungrounded_claims_B": 정수,
  "ungrounded_examples_A": ["구체적 문장 인용", "..."],
  "ungrounded_examples_B": ["구체적 문장 인용", "..."],
  "trustworthy_for_operator": "A" 또는 "B" 또는 "both" 또는 "neither",
  "notes": "자유 기술(선택)"
}}
"""

# Corrected forced-grounding step (2026-11-04). The stage-1 question of INSTRUCTIONS_FULL asked where the WHITE BOX sits in
# the crop, not where the CYAN MARKER sits inside the box, so its answer was never an independent reading of the hotspot
# cell that the reports state. INSTRUCTIONS_FULL2 asks for the marker's cell within the box and is otherwise identical.
INSTRUCTIONS_FULL2 = INSTRUCTIONS_FULL.replace(
    "- 이미지는 열화상 CCTV로 촬영한 설비(철도/발전소/변전소/소각장)의 크롭입니다. 흰 사각형 테두리(마커)가 진단 대상 부품의 위치를 표시합니다.",
    "- 이미지는 열화상 CCTV로 촬영한 설비(철도/발전소/변전소/소각장)의 크롭입니다. 흰 사각형 테두리가 진단 대상 부품(상자)이고, 하늘색 원+십자 표식(마커)이 계측 시스템이 찾은 상자 내 최고온 지점입니다.",
).replace(
    "- 마커(흰 사각형)가 열화상 이미지의 3x3 격자(상단-좌/중/우, 중단-좌/중/우, 하단-좌/중/우) 중 어디에 있습니까?",
    "- 흰 상자 내부를 가로 3칸 × 세로 3칸으로 나눌 때, 하늘색 마커는 어느 칸에 있습니까? (상단-좌/중/우, 중단-좌/중/우, 하단-좌/중/우 중 하나)",
).replace(
    "- 마커가 가리키는 부위는 무엇으로 보입니까(가능하면 가시광도 참고)? 확신 정도는?",
    "- 마커가 놓인 부위는 무엇으로 보입니까(가능하면 가시광도 참고)? 확신 정도는?",
)
assert INSTRUCTIONS_FULL2 != INSTRUCTIONS_FULL and "하늘색 마커는 어느 칸" in INSTRUCTIONS_FULL2

# Protocol knobs (set from the command line or by a task builder):
#   STRIP_TELLS  — remove from BOTH reports the fields whose fixed wording identifies the code-only report
#                  (the '■ 기계 검증' line and the '육안 …/화질 …' tokens of the '■ 이미지 판독' line), so the judge
#                  cannot tell which report is model-written from a template string.
#   FACTS_INLINE — append the case's facts.json (minus file paths) to the prompt, so the judge can check the reports'
#                  position, pattern and session statements against the measurement (the 'judge with facts' protocol).
STRIP_TELLS = False
FACTS_INLINE = False
FACTS_BLOCK = ("\n[계측 사실 — 시스템이 방사 온도 행렬에서 계산한 값(JSON). 보고서의 열점 위치·열 형태·세션 서술을 이 값과 대조해 "
               "근거 없는 주장을 세는 데 사용하세요]\n{facts}\n\n")


def strip_tells(txt):
    import re
    out = []
    for line in txt.splitlines():
        if line.startswith("■ 기계 검증"):
            continue
        if line.startswith("■ 이미지 판독"):
            line = re.sub(r"\s*/ 육안 [^/]*", "", line)
            line = re.sub(r"\s*/ 화질 .*$", "", line)
        out.append(line)
    return "\n".join(out)


def facts_text(case_id):
    f = json.load(open(os.path.join(LOCK, case_id, "facts.json"), encoding="utf-8"))
    f = {k: v for k, v in f.items() if k not in ("images", "frame_key", "paths")}
    return json.dumps(f, ensure_ascii=False)


def load_key(keyfile=None):
    return json.load(open(keyfile or KEYF, encoding="utf-8"))


def report_text(arm, case_id):
    d = V2_DIR if arm == "v2" else A0_DIR
    txt = open(os.path.join(d, case_id + ".txt"), encoding="utf-8").read()
    return strip_tells(txt) if STRIP_TELLS else txt


def set_v2_dir(path):
    """Point the 'v2' arm at another report directory (e.g. the no-image control arm) while keeping the blinding key."""
    global V2_DIR
    V2_DIR = path


def img_path(case_id, name):
    return os.path.join(LOCK, case_id, name)


def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode("ascii")


def build_prompt(case_id, key, schema="simple"):
    a_arm, b_arm = key[case_id]["A"], key[case_id]["B"]
    tmpl = {"full": INSTRUCTIONS_FULL, "full2": INSTRUCTIONS_FULL2}.get(schema, INSTRUCTIONS)
    prompt = tmpl.format(report_a=report_text(a_arm, case_id), report_b=report_text(b_arm, case_id))
    if FACTS_INLINE:   # inserted after .format so the JSON braces are never interpreted
        prompt = prompt.replace("[요청 —", FACTS_BLOCK.format(facts=facts_text(case_id)) + "[요청 —", 1)
    return prompt


def export_packet(cases, key, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for cid in cases:
        cdir = os.path.join(out_dir, cid)
        os.makedirs(cdir, exist_ok=True)
        prompt = build_prompt(cid, key)
        open(os.path.join(cdir, "prompt.txt"), "w", encoding="utf-8").write(prompt)
        for name in ("thermal_crop_v2.png", "rgb_crop_v2.png"):
            src = img_path(cid, name)
            if os.path.exists(src):
                open(os.path.join(cdir, name), "wb").write(open(src, "rb").read())
    readme = (
        "각 하위 폴더(caseXXX/)에 prompt.txt(지시문+두 보고서)와 thermal_crop_v2.png, rgb_crop_v2.png가 있습니다.\n"
        "ChatGPT/Gemini 등 외부 모델 웹 UI에 두 이미지와 prompt.txt 내용을 그대로 붙여넣고,\n"
        "돌아온 JSON 답변을 그 모델명으로 results/judge_crossfamily_<provider>/caseXXX.json 으로 저장한 뒤\n"
        "  python judge_crossfamily.py --mode aggregate --tag <provider>\n"
        "로 집계하세요.\n"
    )
    open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8").write(readme)
    print(f"exported {len(cases)} case packets to {out_dir}")


def call_openai(model, case_id, key):
    from openai import OpenAI
    client = OpenAI()
    prompt = build_prompt(case_id, key)
    content = [{"type": "text", "text": prompt}]
    for name in ("thermal_crop_v2.png", "rgb_crop_v2.png"):
        p = img_path(case_id, name)
        if os.path.exists(p):
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(p)}"}})
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"}, temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def call_gemini(model, case_id, key, schema="simple", temperature=0):
    # Direct REST call (not the google-generativeai SDK): as of mid-2026 Google AI Studio issues
    # "Auth key" format API keys starting with "AQ." (replacing the legacy "AIza..." Standard keys,
    # which stop working entirely in Sept 2026). Older SDK versions client-side-validate the old
    # "AIza" prefix and reject AQ. keys before even sending the request, even though the native
    # REST endpoint accepts AQ. keys fine via the x-goog-api-key header. Calling REST directly
    # sidesteps that SDK-side format check.
    import requests
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    prompt = build_prompt(case_id, key, schema=schema)
    parts = [{"text": prompt}]
    for name in ("thermal_crop_v2.png", "rgb_crop_v2.png"):
        p = img_path(case_id, name)
        if os.path.exists(p):
            parts.append({"inline_data": {"mime_type": "image/png", "data": b64(p)}})
    gen_cfg = {"responseMimeType": "application/json", "temperature": temperature}
    if "lite" not in model:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": gen_cfg,
    }
    r = requests.post(url, headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                       json=body, timeout=240)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    d = r.json()
    text = d["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def run(cases, key, provider, model, out_dir, schema="simple", temperature=0, repeat=1, sleep=0.3):
    os.makedirs(out_dir, exist_ok=True)
    ok, fail = 0, 0
    for cid in cases:
        for rep in range(repeat):
            suffix = "" if repeat == 1 else f"_r{rep}"
            outp = os.path.join(out_dir, cid + suffix + ".json")
            if os.path.exists(outp):
                ok += 1
                continue
            try:
                if provider == "openai":
                    j = call_openai(model, cid, key)
                else:
                    j = call_gemini(model, cid, key, schema=schema, temperature=temperature)
                j["case_id"] = cid
                json.dump(j, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                ok += 1
                print(cid + suffix, "ok")
            except Exception as e:
                fail += 1
                print(cid + suffix, "FAILED:", repr(e)[:160])
                if "429" in repr(e):
                    time.sleep(20)
            time.sleep(sleep)
    print(f"done: {ok} ok, {fail} failed, out_dir={out_dir}")


def aggregate(tag, cases, key):
    in_dir = os.path.join(R, f"judge_crossfamily_{tag}")
    J = []
    for cid in cases:
        fp = os.path.join(in_dir, cid + ".json")
        if not os.path.exists(fp):
            continue
        j = json.load(open(fp, encoding="utf-8"))
        j["case_id"] = cid
        j["A"] = key[cid]["A"]
        j["B"] = key[cid]["B"]
        J.append(j)
    n = len(J)
    if n == 0:
        print(f"no results found in {in_dir} -- nothing to aggregate")
        return
    rng = np.random.default_rng(0)

    def arm(j):
        return "tie" if j["preferred_report"] == "tie" else (j["A"] if j["preferred_report"] == "A" else j["B"])

    def g(j, k, a):
        return j[k + "_A_1to5"] if j["A"] == a else j[k + "_B_1to5"]

    def boot(stat):
        v = [stat([J[i] for i in rng.integers(0, n, n)]) for _ in range(2000)]
        return np.percentile(v, 2.5), np.percentile(v, 97.5)

    pv = lambda items: np.mean([arm(j) == "v2" for j in items])
    du = lambda items: np.mean([g(j, "usefulness", "a0plus") - g(j, "usefulness", "v2") for j in items])
    out = [f"=== 교차계열 판정자({tag}) — 프런티어 VLM(v2) vs A0+(코드전용), 잠금셋 재현 ({n}/{len(cases)}케이스) ==="]
    lo, hi = boot(pv)
    out.append(f"선호: VLM {sum(arm(j)=='v2' for j in J)}/{n} = {pv(J):.3f} [{lo:.3f},{hi:.3f}] | "
                f"A0+ {sum(arm(j)=='a0plus' for j in J)} | tie {sum(arm(j)=='tie' for j in J)}")
    lo, hi = boot(du)
    out.append(f"유용성 차이(A0+ - VLM): {du(J):+.2f} [{lo:+.2f},{hi:+.2f}] "
                f"(VLM {st.mean([g(j,'usefulness','v2') for j in J]):.2f} vs A0+ {st.mean([g(j,'usefulness','a0plus') for j in J]):.2f})")
    which = collections.Counter(j.get("which_report_reads_image_more", "?") for j in J)
    out.append(f"'이미지를 더 읽은 것처럼 보이는 쪽' 판정자 추정: {dict(which)} (실제 VLM arm과 일치 여부는 별도 확인 필요)")
    out.append("")
    out.append("비교 기준선 (동일 32케이스, 같은 계열 판정자=Claude): VLM 선호 21/32 = 0.656 [0.50,0.81]")
    txt = "\n".join(out)
    outp = os.path.join(R, f"judge_crossfamily_{tag}_summary.txt")
    open(outp, "w", encoding="utf-8").write(txt)
    print(txt)
    print("saved", outp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["export", "run", "aggregate"], required=True)
    ap.add_argument("--provider", choices=["openai", "gemini"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N cases (for a quick test)")
    ap.add_argument("--schema", choices=["simple", "full", "full2"], default="simple",
                    help="full = forced-grounding protocol (independent marker/part reading before comparison), matching the original judge_frontier design; "
                         "full2 = the same with the corrected stage-1 question (cyan marker's cell within the box)")
    ap.add_argument("--strip_tells", action="store_true", help="remove the template fields that identify the code-only report from both reports")
    ap.add_argument("--facts", action="store_true", help="append the case's facts.json to the prompt (judge-with-facts protocol)")
    ap.add_argument("--sleep", type=float, default=0.3, help="seconds between API calls (free tier: 15 requests per minute)")
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("--repeat", type=int, default=1, help="run each case this many times (noise estimate); writes caseXXX_r0.json, _r1.json, ...")
    ap.add_argument("--only-new", action="store_true", help="with --key, process only cases absent from the default 32-case key (for incremental expansion)")
    ap.add_argument("--key", default=None, help="path to an alternate blinding key json (default: results/judge_frontier/_key.json)")
    ap.add_argument("--v2_dir", default=None, help="report directory to use as the 'v2' arm (e.g. results/vlm_out/v2frontier/noimg for the no-image control)")
    args = ap.parse_args()
    if args.v2_dir:
        set_v2_dir(args.v2_dir)
    STRIP_TELLS = args.strip_tells
    FACTS_INLINE = args.facts

    key = load_key(args.key)
    cases = sorted(key.keys())
    if args.only_new:
        base_cases = set(load_key().keys())
        cases = [c for c in cases if c not in base_cases]
    if args.limit:
        cases = cases[:args.limit]
    print(f"{len(cases)} locked judging cases selected")

    if args.mode == "export":
        export_packet(cases, key, os.path.join(R, "judge_crossfamily_packet"))
    elif args.mode == "run":
        if not args.provider:
            sys.exit("--provider openai|gemini required for --mode run")
        has_key = (os.environ.get("OPENAI_API_KEY") if args.provider == "openai"
                   else (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")))
        if not has_key:
            sys.exit(f"no API key found in environment for provider={args.provider}. "
                     f"Set OPENAI_API_KEY (openai) or GOOGLE_API_KEY/GEMINI_API_KEY (gemini), or use --mode export instead.")
        model = args.model or ("gpt-4o" if args.provider == "openai" else "gemini-2.5-pro")
        tag = args.tag or args.provider
        run(cases, key, args.provider, model, os.path.join(R, f"judge_crossfamily_{tag}"),
            schema=args.schema, temperature=args.temperature, repeat=args.repeat, sleep=args.sleep)
        if not args.limit and args.repeat == 1:
            aggregate(tag, cases, key)
    elif args.mode == "aggregate":
        if not args.tag:
            sys.exit("--tag required for --mode aggregate (e.g. --tag manual, --tag openai, --tag gemini)")
        aggregate(args.tag, cases, key)
