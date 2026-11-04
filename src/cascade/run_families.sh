#!/bin/sh
cd <PROJECT_ROOT>/src/cascade
PY=python
export HF_HOME='<HF_HOME>'
export PYTHONIOENCODING=utf-8
LOG=<PROJECT_ROOT>/results/vlm_out/families
mkdir -p $LOG
# language ablation on the reference model first (Korean prompt was used everywhere else)
$PY probe_families.py --model Qwen/Qwen2.5-VL-3B-Instruct --lang en > $LOG/qwen25_3b_en.log 2>&1
$PY probe_families.py --model Qwen/Qwen2.5-VL-3B-Instruct --lang ko > $LOG/qwen25_3b_ko.log 2>&1
# other small families (native transformers support, no remote code)
$PY probe_families.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct --lang en > $LOG/smolvlm2_en.log 2>&1
$PY probe_families.py --model OpenGVLab/InternVL3-2B-hf --lang en > $LOG/internvl3_2b_en.log 2>&1
$PY probe_families.py --model Qwen/Qwen2-VL-2B-Instruct --lang en > $LOG/qwen2_2b_en.log 2>&1
$PY probe_families.py --model llava-hf/llava-onevision-qwen2-7b-ov-hf --lang en --quant 4bit > $LOG/llava_ov_7b_en.log 2>&1
$PY probe_families.py --model google/gemma-3-4b-it --lang en > $LOG/gemma3_4b_en.log 2>&1
echo FAMILIES_DONE > $LOG/FAMILIES_DONE
