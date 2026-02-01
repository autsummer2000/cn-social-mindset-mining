"""
社会心理构念生成与评审自动化管道
Author: Your Name
Date: 2026-01-24
Python Version: 3.8+

依赖：
- openai>=1.0.0
- pandas>=1.3.0
- numpy>=1.20.0
- ollama>=0.1.0
- openpyxl>=3.0.0
"""

import os
import csv
import json
import numpy as np
import pandas as pd
from datetime import datetime
from openai import OpenAI
from typing import Dict, List, Tuple
import sys

from prompts_loader import load_prompts
from embeddings_cache import (
    concept_card_text,
    embed_texts_ollama,
    load_or_build_seed_embeddings,
    load_seed_library,
    normalize_rows,
)
from validators import (
    extract_json_from_text,
    parse_construct_with_reason,
    validate_construct_json,
    validate_evaluation_json,
)

# ============================================================================
# 配置区：请根据你的实际情况修改
# ============================================================================

# API 配置
# 统一使用阿里云提供的 QWEN_API_KEY 环境变量（生成器与裁判共用）
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "your-qwen-key-here")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DEEPSEEK_API_KEY = QWEN_API_KEY
DEEPSEEK_BASE_URL = QWEN_BASE_URL

# 生成器（DeepSeek）
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9
GEN_MAX_TOKENS = 2200
GEN_ENABLE_SEARCH = True
GEN_ENABLE_THINKING = True

# 格式修复（DeepSeek）
REPAIR_TEMPERATURE = 0.1
REPAIR_TOP_P = 1.0
REPAIR_MAX_TOKENS = 1800
REPAIR_ENABLE_SEARCH = True
REPAIR_ENABLE_THINKING = False

# judge（Qwen3-Max）
JUDGE_TEMPERATURE = 0.3
JUDGE_TOP_P = 0.5
JUDGE_MAX_TOKENS = 2200
JUDGE_ENABLE_SEARCH = True
JUDGE_SEED = 42  # 设为 None 以避免评分完全固定
QWEN_RESULT_FORMAT = "message"

# 种子库配置
SEED_XLSX = r"D:\博二\AI for science discovery\种子库\种子库_8.xlsx"
SEED_SHEET = "Sheet1"

# Embedding 模型配置（使用本地 Ollama）
EMBED_MODEL = "qwen3-embedding:8b"  # 可选: 0.6b, 4b, 8b
BATCH_SIZE = 64

# 相似度阈值配置
MIN_SIMILARITY = 0.526395499706268  # 最小相似度（太低表示完全不相干）
MAX_SIMILARITY = 0.774457275867462  # 最大相似度（太高表示过于接近）

# 运行控制
MAX_REVISION_ROUNDS = 3  # 最多修订轮数
MAX_FORMAT_REPAIRS = 2   # JSON 格式修复上限
RUNS_PER_EXECUTION = 203   # 单次运行重复流程的次数（可自定义）

# 评审通过标准（更严格）
PASS_MIN_SCORE = 4
PASS_REQUIRE_FIVES = 2

# 判停策略：Hard Limit + Low Yield Stop Loss
MAX_TRIALS = 300
LOW_YIELD_WINDOW = 10
LOW_YIELD_THRESHOLD = 0.1
LOW_YIELD_PATIENCE = 2

# 是否保存相似度全量结果（种子库很大时可关闭）
SAVE_FULL_SIMILARITY = False

# 输出目录（支持实验子目录，避免历史结果混在一起）
EXPERIMENT_TAG = os.getenv("EXPERIMENT_TAG", "").strip()
OUTPUT_BASE_DIR = "./construct_outputs"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, EXPERIMENT_TAG) if EXPERIMENT_TAG else OUTPUT_BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 提示词目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(BASE_DIR, "prompt")

# Embedding 缓存
CACHE_DIR = os.path.join(OUTPUT_DIR, "embedding_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 运行统计输出
TRIAL_METRICS_FILE = os.path.join(OUTPUT_DIR, "trial_metrics.csv")

# 日志文件编码（带 BOM 便于 Windows 工具识别）
LOG_FILE_ENCODING = "utf-8-sig"

# ============================================================================
# 辅助函数
# ============================================================================
def build_format_repair_prompt(template: str, prev_output: str, error_msg: str) -> str:
    """将格式修复模板填充为可用的修复提示词。"""
    return template.format(error_msg=error_msg, prev_output=prev_output)

# ============================================================================
# 日志记录器
# ============================================================================

class Logger:
    """日志记录器"""
    def __init__(self, log_file: str = None):
        self.log_file = log_file
        if log_file:
            with open(log_file, 'w', encoding=LOG_FILE_ENCODING) as f:
                f.write(f"=== 构念生成流程日志 ===\n")
                f.write(f"开始时间: {datetime.now()}\n\n")
    
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"
        print(log_line)
        if self.log_file:
            with open(self.log_file, 'a', encoding=LOG_FILE_ENCODING) as f:
                f.write(log_line + "\n")

def _is_nonempty_str(x: object) -> bool:
    return isinstance(x, str) and x.strip() != ""

def _is_list_of_str(x: object) -> bool:
    return isinstance(x, list) and all(isinstance(i, str) and i.strip() != "" for i in x)

def _meets_strict_pass(qa: Dict) -> bool:
    scores = [
        qa.get("discriminant_validity_score"),
        qa.get("groupness_score"),
        qa.get("stability_score"),
    ]
    if any(not isinstance(s, (int, float)) for s in scores):
        return False
    if min(scores) < PASS_MIN_SCORE:
        return False
    if sum(1 for s in scores if s >= 5) < PASS_REQUIRE_FIVES:
        return False
    return True

def call_deepseek_api(
    messages: List[Dict],
    enable_thinking: bool = True,
    enable_search: bool = True,
    temperature: float = None,
    top_p: float = None,
    max_tokens: int = None,
    logger: Logger = None
) -> Tuple[str, Dict]:
    """
    调用 DeepSeek V3.2 API
    返回: (完整响应文本, 解析后的 JSON 对象)
    """
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    
    if logger:
        logger.log("调用 DeepSeek V3.2 API（思考模式+联网搜索）...")
    
    try:
        # 注意：根据DeepSeek文档，使用extra_body传非标准参数
        params = {
            "model": "deepseek-v3.2",
            "messages": messages,
            "extra_body": {
                "enable_thinking": enable_thinking,
                "enable_search": enable_search
            },
            "stream": False
        }
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        response = client.chat.completions.create(
            **params
        )
        
        # 鑾峰彇鍝嶅簲鍐呭?
        content = response.choices[0].message.content
        
        # 如果有reasoning_content，也记录下来
        reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
        
        full_response = content
        if reasoning:
            full_response = f"[思考过程]\n{reasoning}\n\n[回复内容]\n{content}"
        
        if logger:
            logger.log("DeepSeek API 调用成功", "SUCCESS")
        
        # 鎻愬彇JSON
        json_obj = extract_json_from_text(content, validate_construct_json)
        
        return full_response, json_obj
        
    except Exception as e:
        if logger:
            logger.log(f"DeepSeek API 调用失败: {e}", "ERROR")
        raise

def call_qwen_api(
    messages: List[Dict],
    enable_search: bool = True,
    temperature: float = None,
    top_p: float = None,
    max_tokens: int = None,
    seed: int = None,
    logger: Logger = None
) -> Tuple[str, Dict]:
    """
    调用Qwen3-Max API进行评审
    返回: (完整响应文本, 解析后的JSON对象)
    """
    client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
    
    if logger:
        logger.log("调用 Qwen3-Max API 进行评审...")
    
    try:
        params = {
            "model": "qwen3-max",
            "messages": messages,
            "extra_body": {
                "enable_search": enable_search,
                "result_format": QWEN_RESULT_FORMAT
            },
            "stream": False
        }
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if seed is not None:
            params["seed"] = seed

        response = client.chat.completions.create(**params)
        
        content = response.choices[0].message.content
        
        if logger:
            logger.log("Qwen3-Max API 调用成功", "SUCCESS")
        
        # 鎻愬彇JSON
        json_obj = extract_json_from_text(content, validate_evaluation_json)
        
        return content, json_obj
        
    except Exception as e:
        if logger:
            logger.log(f"Qwen3-Max API 调用失败: {e}", "ERROR")
        raise

# ============================================================================
# 鐩镐技搴︽?鏌?
# ============================================================================

def check_similarity(
    construct: Dict,
    seed_df: pd.DataFrame,
    seed_embeddings: np.ndarray,
    embed_model: str,
    logger: Logger = None
) -> Tuple[bool, Dict]:
    """
    检查生成构念与种子库的相似度
    返回: (是否通过, 相似度信息)
    """
    if logger:
        logger.log("\n[4.3] 相似度检查...")
    
    # 构?念卡片文?
    gen_text = concept_card_text(
        construct.get("construct_name", ""),
        construct.get("definition", "")
    )
    
    # 鐢熸垚embedding
    gen_emb = embed_texts_ollama([gen_text], model=embed_model, batch_size=1)[0]
    gen_emb = normalize_rows(gen_emb.reshape(1, -1))[0]
    
    # 计算相似度
    similarities = seed_embeddings.dot(gen_emb)
    
    # 找到相似的Top-K
    result_df = seed_df.copy()
    result_df["similarity"] = similarities
    result_df = result_df.sort_values("similarity", ascending=False).reset_index(drop=True)
    
    # 获取大相似度
    max_sim = float(result_df.loc[0, "similarity"])
    
    # 鍒ゆ柇
    if max_sim > MAX_SIMILARITY:
        verdict = f"过于相似 (大相似度 {max_sim:.4f} > 阈值{MAX_SIMILARITY})"
        passed = False
    elif max_sim < MIN_SIMILARITY:
        verdict = f"过于不相似 (最大相似度 {max_sim:.4f} < 阈值 {MIN_SIMILARITY})"
        passed = False
    else:
        verdict = f"相似度通过（范围 [{MIN_SIMILARITY}, {MAX_SIMILARITY}]）"
        passed = True
    
    if logger:
        logger.log(f"相似度检查结果: {verdict}", "INFO" if passed else "WARNING")
        logger.log(f"生成构念大相似度: {max_sim:.4f}")
    
    # 鎻愬彇Top-5
    top5 = result_df.head(5)[["类型", "词名", "定义", "similarity"]].to_dict('records')
    
    similarity_info = {
        "max_similarity": max_sim,
        "verdict": verdict,
        "passed": passed,
        "top5_similar": top5,
        "full_results": result_df.to_dict('records') if SAVE_FULL_SIMILARITY else None
    }
    
    return passed, similarity_info

# ============================================================================
# 涓绘祦绋?
# ============================================================================

def run_construct_generation_pipeline(
    run_id: str = None,
    logger: Logger = None
) -> Dict:
    """
    执行完整的构念生成与评审流程
    返回: 包含元数据的字典
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    if logger is None:
        log_file = os.path.join(OUTPUT_DIR, f"log_{run_id}.txt")
        logger = Logger(log_file)
    
    # 加载提示词（便于后续直接?prompt 文件而不用改代码?
    generator_prompt, judge_prompt_template, revision_prompt_template, format_repair_template = load_prompts(PROMPT_DIR)

    logger.log("=" * 80)
    logger.log("开始构念生成流程")
    logger.log(f"运行ID: {run_id}")
    logger.log("=" * 80)
    
    # 1. 加载种子?
    logger.log("\n[步骤1] 加载种子库...")
    seed_df = load_seed_library(SEED_XLSX, SEED_SHEET)
    logger.log(f"种子库加载完成，共{len(seed_df)} 条")
    
    # 2. 生成种子库embeddings（支持缓存?量更新）
    logger.log("\n[步骤2] 生成种子库向量...")
    seed_embeddings = load_or_build_seed_embeddings(
        seed_df=seed_df,
        embed_model=EMBED_MODEL,
        batch_size=BATCH_SIZE,
        cache_dir=CACHE_DIR,
        seed_xlsx=SEED_XLSX,
        logger=logger
    )
    seed_embeddings = normalize_rows(seed_embeddings)
    logger.log(f"种子库向量生成完成，维度: {seed_embeddings.shape}")
    
    # 3. 初?化结果??
    pipeline_result = {
        "run_id": run_id,
        "start_time": datetime.now().isoformat(),
        "config": {
            "min_similarity": MIN_SIMILARITY,
            "max_similarity": MAX_SIMILARITY,
            "max_rounds": MAX_REVISION_ROUNDS,
            "embed_model": EMBED_MODEL
        },
        "rounds": [],
        "reject_reason_counts": {},
        "final_status": None,
        "final_construct": None
    }
    
    # 4. 始迭代生成与评审
    current_round = 1
    previous_construct = None
    previous_feedback = None
    
    while current_round <= MAX_REVISION_ROUNDS:
        logger.log("\n" + "=" * 80)
        logger.log(f"[第{current_round} 轮]")
        logger.log("=" * 80)
        
        round_data = {
            "round": current_round,
            "deepseek_output": None,
            "similarity_check": None,
            "qwen_output": None,
            "format_repairs": 0,
            "format_errors": [],
            "reject_reason": None,
            "status": None
        }
        
        # 4.1 调用DeepSeek生成构念
        logger.log("\n[4.1] 调用DeepSeek生成构念...")
        
        if current_round == 1:
            # ??使用原?提示?
            messages = [{"role": "user", "content": generator_prompt}]
        else:
            # 鍚庣画杞??锛氬寘鍚?慨璁㈡寚浠?
            revision_prompt = revision_prompt_template.replace(
                "{judge_output}",
                json.dumps(previous_feedback, ensure_ascii=False, indent=2)
            )
            messages = [
                {"role": "user", "content": generator_prompt},
                {"role": "assistant", "content": json.dumps(previous_construct, ensure_ascii=False)},
                {"role": "user", "content": revision_prompt}
            ]
        
        try:
            deepseek_response, construct_json = call_deepseek_api(
                messages=messages,
                enable_thinking=GEN_ENABLE_THINKING,
                enable_search=GEN_ENABLE_SEARCH,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                max_tokens=GEN_MAX_TOKENS,
                logger=logger
            )
            
            round_data["deepseek_output"] = {
                "raw_response": deepseek_response,
                "parsed_json": construct_json,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.log(f"DeepSeek 调用失败，终止流程: {e}", "ERROR")
            round_data["status"] = "DEEPSEEK_ERROR"
            round_data["reject_reason"] = "DEEPSEEK_ERROR"
            pipeline_result["rounds"].append(round_data)
            pipeline_result["reject_reason_counts"][round_data["reject_reason"]] = pipeline_result["reject_reason_counts"].get(round_data["reject_reason"], 0) + 1
            pipeline_result["final_status"] = "FAILED"
            break
        
        # 4.2 校验输出格式（失败则进入格式??回路?
        logger.log("\n[4.2] 校验输出格式...")
        if construct_json is None:
            logger.log("无法从 DeepSeek 响应中提取有效 JSON，进入格式修复流程...", "WARNING")
            repair_count = 0
            while repair_count < MAX_FORMAT_REPAIRS and construct_json is None:
                repair_count += 1
                round_data["format_repairs"] = repair_count
                _, error_msg = parse_construct_with_reason(deepseek_response)
                round_data["format_errors"].append(error_msg)
                repair_prompt = build_format_repair_prompt(format_repair_template, deepseek_response, error_msg)
                try:
                    deepseek_response, construct_json = call_deepseek_api(
                        messages=[{"role": "user", "content": repair_prompt}],
                        enable_thinking=REPAIR_ENABLE_THINKING,
                        enable_search=REPAIR_ENABLE_SEARCH,
                        temperature=REPAIR_TEMPERATURE,
                        top_p=REPAIR_TOP_P,
                        max_tokens=REPAIR_MAX_TOKENS,
                        logger=logger
                    )
                    round_data["deepseek_output"] = {
                        "raw_response": deepseek_response,
                        "parsed_json": construct_json,
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.log(f"格式修复调用失败: {e}", "ERROR")
                    construct_json = None
            if construct_json is None:
                logger.log("格式修复失败，淘汰该候选", "ERROR")
                round_data["status"] = "INVALID_FORMAT"
                round_data["reject_reason"] = "FORMAT_FAIL"
                pipeline_result["rounds"].append(round_data)
                pipeline_result["reject_reason_counts"][round_data["reject_reason"]] = pipeline_result["reject_reason_counts"].get(round_data["reject_reason"], 0) + 1
                current_round += 1
                continue
        
        logger.log("输出格式校验通过 ✓", "SUCCESS")
        logger.log(f"构念名称: {construct_json['construct_name']}")
        
        # 4.3 相似度?查（不过关直接淘汰，避免多余评审?
        similarity_passed, similarity_info = check_similarity(
            construct=construct_json,
            seed_df=seed_df,
            seed_embeddings=seed_embeddings,
            embed_model=EMBED_MODEL,
            logger=logger
        )
        
        round_data["similarity_check"] = similarity_info
        
        if not similarity_passed:
            logger.log("相似度检查未通过，淘汰该构念", "WARNING")
            round_data["status"] = "SIMILARITY_REJECTED"
            if similarity_info["max_similarity"] > MAX_SIMILARITY:
                round_data["reject_reason"] = "SIM_TOO_HIGH"
            elif similarity_info["max_similarity"] < MIN_SIMILARITY:
                round_data["reject_reason"] = "SIM_TOO_LOW"
            else:
                round_data["reject_reason"] = "SIMILARITY_REJECTED"
            pipeline_result["rounds"].append(round_data)
            pipeline_result["reject_reason_counts"][round_data["reject_reason"]] = pipeline_result["reject_reason_counts"].get(round_data["reject_reason"], 0) + 1
            pipeline_result["final_status"] = "SIMILARITY_REJECTED"
            break
        
        logger.log("相似度检查通过 ✓", "SUCCESS")
        
        # 4.4 璋冪敤Qwen璇勫?
        logger.log("\n[4.4] 调用 Qwen3-Max 评审...")
        
        # 构造评审提示词：基础模板 + （若有）上一轮反馈与上一版草案，要求先逐条验收再评分
        judge_prompt = judge_prompt_template.replace(
            "{input_construct}",
            json.dumps(construct_json, ensure_ascii=False, indent=2)
        )
        if previous_feedback is not None:
            verification_block = (
                "\n\n【上一轮评审反馈（JSON）】\n" + json.dumps(previous_feedback, ensure_ascii=False, indent=2) +
                "\n\n【上一轮候选构念（JSON）】\n" + json.dumps(previous_construct, ensure_ascii=False, indent=2) +
                "\n\n【核验要求】请先逐条核对上一轮 revision_patch_list/major_flaw 是否已在本轮修订稿中落实，再进行本轮打分与给出新的 revision_patch_list。务必仅输出纯 JSON，不要额外文字。"
            )
            judge_prompt = judge_prompt + verification_block
        # 可选：将本轮评审提示词落盘，方便排查“分数不变”的问题
        try:
            jp_path = os.path.join(OUTPUT_DIR, f"judge_prompt_{run_id}_r{current_round}.txt")
            with open(jp_path, 'w', encoding='utf-8') as f:
                f.write(judge_prompt)
        except Exception:
            pass
        try:
            qwen_response, evaluation_json = call_qwen_api(
                messages=[{"role": "user", "content": judge_prompt}],
                enable_search=JUDGE_ENABLE_SEARCH,
                temperature=JUDGE_TEMPERATURE,
                top_p=JUDGE_TOP_P,
                max_tokens=JUDGE_MAX_TOKENS,
                seed=JUDGE_SEED,
                logger=logger
            )
            
            round_data["qwen_output"] = {
                "raw_response": qwen_response,
                "parsed_json": evaluation_json,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.log(f"Qwen 调用失败，终止流程: {e}", "ERROR")
            round_data["status"] = "QWEN_ERROR"
            round_data["reject_reason"] = "QWEN_ERROR"
            pipeline_result["rounds"].append(round_data)
            pipeline_result["reject_reason_counts"][round_data["reject_reason"]] = pipeline_result["reject_reason_counts"].get(round_data["reject_reason"], 0) + 1
            pipeline_result["final_status"] = "FAILED"
            break
        
        # 4.5 判断评审结果
        if evaluation_json is None:
            logger.log("无法从 Qwen 响应中提取有效 JSON，进入评审格式修复重试...", "WARNING")
            repair_prompt = (
                "你的上一条评审输出无法被解析为 JSON。"
                "请严格只输出一个纯 JSON 对象，不要任何解释或额外文字。"
                "务必符合评审输出格式要求。"
            )
            try:
                qwen_response, evaluation_json = call_qwen_api(
                    messages=[{"role": "user", "content": repair_prompt}],
                    enable_search=JUDGE_ENABLE_SEARCH,
                    temperature=JUDGE_TEMPERATURE,
                    top_p=JUDGE_TOP_P,
                    max_tokens=JUDGE_MAX_TOKENS,
                    seed=JUDGE_SEED,
                    logger=logger
                )
                round_data["qwen_output"] = {
                    "raw_response": qwen_response,
                    "parsed_json": evaluation_json,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.log(f"评审格式修复失败: {e}", "ERROR")
                evaluation_json = None

        if evaluation_json is None:
            logger.log("评审输出仍无法解析，结束本次轮次（不推进修订轮）", "ERROR")
            round_data["status"] = "INVALID_EVALUATION"
            round_data["reject_reason"] = "INVALID_EVALUATION"
            pipeline_result["rounds"].append(round_data)
            pipeline_result["reject_reason_counts"][round_data["reject_reason"]] = pipeline_result["reject_reason_counts"].get(round_data["reject_reason"], 0) + 1
            pipeline_result["final_status"] = "INVALID_EVALUATION"
            break
        
        status = evaluation_json.get("status", "").upper()
        qa = evaluation_json.get("quantitative_assessment", {})
        if logger and qa:
            logger.log(
                f"评审打分: 区分效度={qa.get('discriminant_validity_score')}, "
                f"群体性={qa.get('groupness_score')}, 稳定性={qa.get('stability_score')}"
            )

        strict_pass = _meets_strict_pass(qa)
        if status == "PASS" and not strict_pass:
            logger.log("评审给出 PASS 但未满足严格通过标准，转为 NEEDS_REVISION", "WARNING")
            status = "NEEDS_REVISION"
            evaluation_json["status"] = "NEEDS_REVISION"
            # 补齐最小反馈结构，避免后续修订缺失
            if "qualitative_critique" not in evaluation_json:
                evaluation_json["qualitative_critique"] = {
                    "major_flaw": "评分未达到严格通过标准，需要进一步提升关键维度质量。",
                    "logic_gap": "至少有一项评分未达到 5 分，请针对薄弱维度加强论证与边界。",
                }
            if not evaluation_json.get("revision_patch_list"):
                patch_list = []
                score_map = {
                    "discriminant_validity_score": "discriminant_validity",
                    "groupness_score": "group_attributes",
                    "stability_score": "emergence_period",
                }
                for k, target in score_map.items():
                    if qa.get(k, 0) < 5:
                        patch_list.append({
                            "priority": "HIGH",
                            "target_field": target,
                            "action_item": "针对该维度补充更具区分力/群体性/稳定性的证据与论证。",
                            "acceptance_criteria": "该维度评分达到 5 分。",
                        })
                evaluation_json["revision_patch_list"] = patch_list or [{
                    "priority": "MEDIUM",
                    "target_field": "definition",
                    "action_item": "整体提升定义的清晰度与可检验性。",
                    "acceptance_criteria": "至少两项评分达到 5 分，且三项均≥4。",
                }]
            if round_data.get("qwen_output") is not None:
                round_data["qwen_output"]["parsed_json"] = evaluation_json
        
        if status == "PASS":
            logger.log(f"\n🎉 第{current_round} 轮评审通过", "SUCCESS")
            round_data["status"] = "PASS"
            pipeline_result["rounds"].append(round_data)
            pipeline_result["reject_reason_counts"]["PASS"] = pipeline_result["reject_reason_counts"].get("PASS", 0) + 1
            pipeline_result["final_status"] = "PASS"
            pipeline_result["final_construct"] = construct_json
            pipeline_result["pass_round"] = current_round
            
            # 保存到?子库
            save_to_seed_library(
                construct=construct_json,
                similarity_info=similarity_info,
                evaluation=evaluation_json,
                round_num=current_round,
                run_id=run_id,
                logger=logger
            )
            
            break
        
        elif status == "NEEDS_REVISION":
            logger.log(f"第{current_round} 轮需要修订", "WARNING")
            round_data["status"] = "NEEDS_REVISION"
            round_data["reject_reason"] = "NEEDS_REVISION"
            pipeline_result["rounds"].append(round_data)
            pipeline_result["reject_reason_counts"][round_data["reject_reason"]] = pipeline_result["reject_reason_counts"].get(round_data["reject_reason"], 0) + 1
            
            if current_round == MAX_REVISION_ROUNDS:
                logger.log("已达最大修订轮次，淘汰该构念", "ERROR")
                pipeline_result["final_status"] = "MAX_ROUNDS_EXCEEDED"
                break
            
            # 保存当前构念和反馈，用于下一?
            previous_construct = construct_json
            previous_feedback = evaluation_json
            
        else:
            logger.log(f"未知评审状态: {status}", "WARNING")
            round_data["status"] = "UNKNOWN_STATUS"
            round_data["reject_reason"] = "UNKNOWN_STATUS"
            pipeline_result["rounds"].append(round_data)
            pipeline_result["reject_reason_counts"][round_data["reject_reason"]] = pipeline_result["reject_reason_counts"].get(round_data["reject_reason"], 0) + 1
        
        current_round += 1
    
    # 5. 保存完整流程记录
    pipeline_result["end_time"] = datetime.now().isoformat()
    
    result_file = os.path.join(OUTPUT_DIR, f"pipeline_result_{run_id}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(pipeline_result, f, ensure_ascii=False, indent=2)

    logger.log(f"\n流程完成，结果已保存至 {result_file}")
    logger.log(f"最终状态: {pipeline_result['final_status']}")

    # 输出每轮摘要与试次统计，便于后续分析/画图
    _write_rounds_summary(pipeline_result)
    _append_trial_metrics(pipeline_result)
    
    return pipeline_result

def run_multiple_pipelines(num_runs: int) -> List[Dict]:
    """连续运行多次完整流程，返回结果列表并汇总统计。"""
    results = []
    summary_log_path = os.path.join(
        OUTPUT_DIR, f"multi_run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    summary_logger = Logger(summary_log_path)
    summary_logger.log(f"多次运行开始：计划执行 {num_runs} 次试次")
    accept_history = []
    low_yield_streak = 0
    for i in range(num_runs):
        if i >= MAX_TRIALS:
            print(f"\n达到 Hard Limit：已执行 {MAX_TRIALS} 次试次，停止运行。")
            summary_logger.log(
                f"触发 Hard Limit：已执行 {MAX_TRIALS} 次试次，停止运行。", "WARNING"
            )
            break

        print(f"\n=== 执行第 {i + 1}/{num_runs} 次 ===")
        result = run_construct_generation_pipeline()
        results.append(result)

        accepted = 1 if result.get("final_status") == "PASS" else 0
        accept_history.append(accepted)

        if len(accept_history) >= LOW_YIELD_WINDOW:
            window = accept_history[-LOW_YIELD_WINDOW:]
            p_hat = sum(window) / LOW_YIELD_WINDOW
            if p_hat < LOW_YIELD_THRESHOLD:
                low_yield_streak += 1
            else:
                low_yield_streak = 0

            if low_yield_streak >= LOW_YIELD_PATIENCE:
                print(
                    f"\n触发 Low Yield Stop Loss：连续 {LOW_YIELD_PATIENCE} 个窗口"
                    f"的边际产出率 < {LOW_YIELD_THRESHOLD}，停止运行。"
                )
                summary_logger.log(
                    f"触发 Low Yield Stop Loss：p_hat={p_hat:.3f}，"
                    f"连续 {LOW_YIELD_PATIENCE} 个窗口低于阈值 {LOW_YIELD_THRESHOLD}，停止运行。",
                    "WARNING",
                )
                break
    _write_runs_summary(results)
    return results

def _write_runs_summary(results: List[Dict]) -> None:
    """将多次运行的汇总结果写入 construct_outputs。"""
    summary = {
        "total_runs": len(results),
        "pass_count": sum(1 for r in results if r.get("final_status") == "PASS"),
        "final_status_counts": {},
        "reject_reason_counts": {},
        "runs": []
    }
    for r in results:
        status = r.get("final_status")
        summary["final_status_counts"][status] = summary["final_status_counts"].get(status, 0) + 1
        for k, v in (r.get("reject_reason_counts") or {}).items():
            summary["reject_reason_counts"][k] = summary["reject_reason_counts"].get(k, 0) + v
        run_id = r.get("run_id")
        summary["runs"].append({
            "run_id": run_id,
            "final_status": status,
            "result_path": os.path.join(OUTPUT_DIR, f"pipeline_result_{run_id}.json")
        })

    out_path = os.path.join(OUTPUT_DIR, "runs_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _append_trial_metrics(pipeline_result: Dict) -> None:
    """追加单次运行的试次统计，便于画累计产出与边际产出曲线。"""
    path = TRIAL_METRICS_FILE
    exists = os.path.exists(path)
    # 计算当前试次序号（不含表头）
    trial_index = 1
    if exists:
        with open(path, "r", encoding=LOG_FILE_ENCODING) as f:
            trial_index = max(1, sum(1 for _ in f) - 1 + 1)

    final_status = pipeline_result.get("final_status")
    accepted = 1 if final_status == "PASS" else 0

    row = {
        "trial_index": trial_index,
        "run_id": pipeline_result.get("run_id"),
        "final_status": final_status,
        "pass_round": pipeline_result.get("pass_round"),
        "accepted": accepted,
        "timestamp": pipeline_result.get("end_time"),
    }

    fieldnames = ["trial_index", "run_id", "final_status", "pass_round", "accepted", "timestamp"]
    with open(path, "a", encoding=LOG_FILE_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_rounds_summary(pipeline_result: Dict) -> None:
    """将每一轮的生成、相似度、裁判反馈摘要写入 CSV 方便分析。"""
    run_id = pipeline_result.get("run_id")
    out_path = os.path.join(OUTPUT_DIR, f"rounds_summary_{run_id}.csv")

    fieldnames = [
        "run_id",
        "round",
        "round_status",
        "reject_reason",
        "format_repairs",
        "construct_name",
        "deepseek_raw",
        "deepseek_json",
        "max_similarity",
        "similarity_passed",
        "similarity_verdict",
        "qwen_status",
        "score_discriminant",
        "score_groupness",
        "score_stability",
        "major_flaw",
        "logic_gap",
        "revision_patch_list",
    ]

    with open(out_path, "w", encoding=LOG_FILE_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in pipeline_result.get("rounds", []):
            deepseek = r.get("deepseek_output") or {}
            similarity = r.get("similarity_check") or {}
            qwen = r.get("qwen_output") or {}
            qwen_parsed = qwen.get("parsed_json") or {}
            qa = qwen_parsed.get("quantitative_assessment") or {}
            qc = qwen_parsed.get("qualitative_critique") or {}

            deepseek_json = deepseek.get("parsed_json")
            construct_name = None
            if isinstance(deepseek_json, dict):
                construct_name = deepseek_json.get("construct_name")

            row = {
                "run_id": run_id,
                "round": r.get("round"),
                "round_status": r.get("status"),
                "reject_reason": r.get("reject_reason"),
                "format_repairs": r.get("format_repairs"),
                "construct_name": construct_name,
                "deepseek_raw": deepseek.get("raw_response"),
                "deepseek_json": json.dumps(deepseek_json, ensure_ascii=False),
                "max_similarity": similarity.get("max_similarity"),
                "similarity_passed": similarity.get("passed"),
                "similarity_verdict": similarity.get("verdict"),
                "qwen_status": qwen_parsed.get("status"),
                "score_discriminant": qa.get("discriminant_validity_score"),
                "score_groupness": qa.get("groupness_score"),
                "score_stability": qa.get("stability_score"),
                "major_flaw": qc.get("major_flaw"),
                "logic_gap": qc.get("logic_gap"),
                "revision_patch_list": json.dumps(qwen_parsed.get("revision_patch_list"), ensure_ascii=False),
            }
            writer.writerow(row)

# ============================================================================
# 保存到?子库
# ============================================================================

# ============================================================================
# 保存到种子库
# ============================================================================

def save_to_seed_library(
    construct: Dict,
    similarity_info: Dict,
    evaluation: Dict,
    round_num: int,
    run_id: str,
    logger: Logger
):
    """将通过评审的构念保存到种子库。"""
    logger.log("\n[保存] 将构念添加到种子库...")

    new_entry = {
        "类型": "AI生成",
        "词名": construct["construct_name"],
        "定义": construct["definition"],
        "完整构念": json.dumps(construct, ensure_ascii=False),
        "修订轮次": round_num,
        "评分_区分效度": evaluation["quantitative_assessment"]["discriminant_validity_score"],
        "评分_群体性": evaluation["quantitative_assessment"]["groupness_score"],
        "评分_稳定性": evaluation["quantitative_assessment"]["stability_score"],
        "最大相似度": similarity_info["max_similarity"],
        "Top5相似概念": json.dumps(similarity_info["top5_similar"], ensure_ascii=False),
        "生成时间": datetime.now().isoformat(),
        "运行ID": run_id,
        "embedding模型": EMBED_MODEL,
    }

    try:
        seed_df = pd.read_excel(SEED_XLSX, sheet_name=SEED_SHEET)
    except Exception as e:
        logger.log(f"读取种子库失败: {e}", "ERROR")
        return

    new_df = pd.DataFrame([new_entry])
    updated_df = pd.concat([seed_df, new_df], ignore_index=True)

    backup_file = SEED_XLSX.replace(".xlsx", f"_backup_{run_id}.xlsx")
    try:
        seed_df.to_excel(backup_file, index=False)
        logger.log(f"原始种子库已备份至: {backup_file}")

        updated_df.to_excel(SEED_XLSX, index=False)
        logger.log(f"新构念已添加到种子库: {SEED_XLSX}", "SUCCESS")

    except Exception as e:
        logger.log(f"保存种子库失败: {e}", "ERROR")

# ============================================================================
# 运行入口
# ============================================================================

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    """程序入口"""
    print("=" * 80)
    print("社会心理构念生成与评审自动化管道")
    print("Construct Generation & Review Pipeline")
    print("=" * 80)

    print("\n[配置检查]")
    print(f"DeepSeek API Key: {'configured' if DEEPSEEK_API_KEY != 'your-deepseek-key-here' else 'not configured'}")
    print(f"Qwen API Key: {'configured' if QWEN_API_KEY != 'your-qwen-key-here' else 'not configured'}")
    print(f"种子库路径: {SEED_XLSX}")
    print(f"Embedding模型: {EMBED_MODEL}")
    print(f"相似度范围: [{MIN_SIMILARITY}, {MAX_SIMILARITY}]")
    print(f"最大修订轮数: {MAX_REVISION_ROUNDS}")
    print(f"输出目录: {OUTPUT_DIR}")

    if not os.path.exists(SEED_XLSX):
        print(f"\n错误：种子库文件不存在 {SEED_XLSX}")
        return

    print("\n按 Enter 开始执行，Ctrl+C 取消...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n已取消")
        return

    try:
        if RUNS_PER_EXECUTION <= 1:
            result = run_construct_generation_pipeline()
        else:
            results = run_multiple_pipelines(RUNS_PER_EXECUTION)
            result = results[-1]

        print("\n" + "=" * 80)
        print("流程执行完成")
        print("=" * 80)
        print(f"\n最终状态: {result['final_status']}")

        if result['final_status'] == 'PASS':
            print("构念已通过评审并保存到种子库")
            print(f"构念名称: {result['final_construct']['construct_name']}")
            print(f"修订轮次: {result['pass_round']}")
        elif result['final_status'] == 'SIMILARITY_REJECTED':
            print("构念因相似度不符合要求而被淘汰")
        elif result['final_status'] == 'MAX_ROUNDS_EXCEEDED':
            print(f"构念在 {MAX_REVISION_ROUNDS} 轮修订后仍未通过")
        else:
            print("流程异常结束")

        print(f"\n详细结果请查看: {OUTPUT_DIR}")

    except Exception as e:
        print(f"\n运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
