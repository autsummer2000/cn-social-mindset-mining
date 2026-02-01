import os
from typing import Tuple

DEFAULT_PROMPT_DIR = "prompt"
GENERATOR_PROMPT_FILENAME = "提示词_生成.txt"
JUDGE_PROMPT_FILENAME = "提示词_判断.txt"
REVISION_PROMPT_FILENAME = "提示词_返回.txt"
FORMAT_REPAIR_PROMPT_FILENAME = "提示词_格式修复.txt"


def _load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_prompts(prompt_dir: str = None) -> Tuple[str, str, str, str]:
    base_dir = prompt_dir or DEFAULT_PROMPT_DIR
    generator_path = os.path.join(base_dir, GENERATOR_PROMPT_FILENAME)
    judge_path = os.path.join(base_dir, JUDGE_PROMPT_FILENAME)
    revision_path = os.path.join(base_dir, REVISION_PROMPT_FILENAME)
    format_repair_path = os.path.join(base_dir, FORMAT_REPAIR_PROMPT_FILENAME)
    return (
        _load_prompt(generator_path),
        _load_prompt(judge_path),
        _load_prompt(revision_path),
        _load_prompt(format_repair_path),
    )
