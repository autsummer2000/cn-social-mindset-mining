from typing import Dict, Optional, Tuple
import json
import re


def _is_nonempty_str(x: object) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _is_list_of_str(x: object) -> bool:
    return isinstance(x, list) and all(isinstance(i, str) and i.strip() != "" for i in x)


CONSTRUCT_REQUIRED_FIELDS = [
    "construct_name",
    "definition",
    "emergence_period",
    "group_attributes",
    "discriminant_validity",
    "linguistic_indicators",
    "predictive_validity",
]


def validate_construct_json(obj: object) -> bool:
    """校验生成构念 JSON 的结构与字段完整性。"""
    if not isinstance(obj, dict):
        return False
    if any(k not in obj for k in CONSTRUCT_REQUIRED_FIELDS):
        return False
    if not _is_nonempty_str(obj["construct_name"]):
        return False
    if not _is_nonempty_str(obj["definition"]):
        return False
    if not _is_nonempty_str(obj["emergence_period"]):
        return False
    if not _is_nonempty_str(obj["group_attributes"]):
        return False

    dv = obj.get("discriminant_validity")
    if not isinstance(dv, dict):
        return False
    for key in ["concept_a", "concept_b"]:
        if key not in dv or not isinstance(dv[key], dict):
            return False
        if not _is_nonempty_str(dv[key].get("concept_name", "")):
            return False
        if not _is_nonempty_str(dv[key].get("core_difference", "")):
            return False
    if not _is_nonempty_str(dv.get("unique_explanatory_power", "")):
        return False

    li = obj.get("linguistic_indicators")
    if not isinstance(li, dict):
        return False
    if not _is_list_of_str(li.get("keywords", [])):
        return False
    if not _is_nonempty_str(li.get("typical_expression_pattern", "")):
        return False

    pv = obj.get("predictive_validity")
    if not isinstance(pv, dict):
        return False
    if not _is_nonempty_str(pv.get("predictable_behaviors", "")):
        return False
    if not _is_nonempty_str(pv.get("consumption_preference", "")):
        return False
    return True


def validate_evaluation_json(obj: object) -> bool:
    """校验评审 JSON 的结构与字段完整性。"""
    if not isinstance(obj, dict):
        return False
    status = obj.get("status")
    if status not in {"PASS", "NEEDS_REVISION"}:
        return False
    qa = obj.get("quantitative_assessment")
    if not isinstance(qa, dict):
        return False
    for k in ["discriminant_validity_score", "groupness_score", "stability_score"]:
        v = qa.get(k)
        if not isinstance(v, (int, float)) or not (1 <= v <= 5):
            return False
    if status == "NEEDS_REVISION":
        qc = obj.get("qualitative_critique")
        if not isinstance(qc, dict):
            return False
        if not _is_nonempty_str(qc.get("major_flaw", "")):
            return False
        if not _is_nonempty_str(qc.get("logic_gap", "")):
            return False
        rpl = obj.get("revision_patch_list")
        if not isinstance(rpl, list) or len(rpl) == 0:
            return False
        for item in rpl:
            if not isinstance(item, dict):
                return False
            if item.get("priority") not in {"HIGH", "MEDIUM", "LOW"}:
                return False
            if not _is_nonempty_str(item.get("target_field", "")):
                return False
            if not _is_nonempty_str(item.get("action_item", "")):
                return False
            if not _is_nonempty_str(item.get("acceptance_criteria", "")):
                return False
    return True


def parse_construct_with_reason(text: str) -> Tuple[Optional[Dict], str]:
    """解析生成构念 JSON，并返回失败原因（用于格式修复提示）。"""
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if validate_construct_json(obj):
            return obj, ""
        if isinstance(obj, dict):
            missing = [k for k in CONSTRUCT_REQUIRED_FIELDS if k not in obj]
            if missing:
                return None, f"字段缺失: {missing}"
        return None, "JSON字段不符合要求"
    except json.JSONDecodeError:
        pass

    json_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for m in matches:
        try:
            obj = json.loads(m)
            if validate_construct_json(obj):
                return obj, ""
            if isinstance(obj, dict):
                missing = [k for k in CONSTRUCT_REQUIRED_FIELDS if k not in obj]
                if missing:
                    return None, f"字段缺失: {missing}"
            return None, "JSON字段不符合要求"
        except json.JSONDecodeError:
            continue

    return None, "JSON解析失败"


def extract_json_from_text(text: str, validator) -> Optional[Dict]:
    """从文本中提取并校验JSON对象"""
    try:
        obj = json.loads(text.strip())
        if validator(obj):
            return obj
    except json.JSONDecodeError:
        pass

    json_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for m in matches:
        try:
            obj = json.loads(m)
            if validator(obj):
                return obj
        except json.JSONDecodeError:
            continue

    return None
