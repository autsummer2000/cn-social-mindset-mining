import hashlib
import inspect
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import ollama


PREPROCESS_VERSION = "concept_card_text_v1"


def concept_card_text(term: str, definition: str) -> str:
    """把概念转成统一文本表示（概念卡片）"""
    term = (term or "").strip()
    definition = (definition or "").strip() if definition else ""
    if definition:
        return f"概念：{term}。定义：{definition}"
    return f"概念：{term}。"


def preprocess_fingerprint() -> str:
    """对预处理函数源码取哈希，用于缓存版本判断。"""
    try:
        src = inspect.getsource(concept_card_text)
        return hashlib.sha256(src.encode("utf-8")).hexdigest()
    except Exception:
        return PREPROCESS_VERSION


def load_seed_library(xlsx_path: str, sheet: str) -> pd.DataFrame:
    """读取并清洗种子库"""
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    required_cols = {"类型", "词名", "定义"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"种子库缺少列：{missing}；当前列：{list(df.columns)}")
    df["类型"] = df["类型"].astype(str).str.strip()
    df["词名"] = df["词名"].astype(str).str.strip()
    df["定义"] = df["定义"].astype(str).str.strip()
    df = df[df["词名"].notna() & (df["词名"] != "")]
    return df.reset_index(drop=True)


def embed_texts_ollama(texts: List[str], model: str, batch_size: int = 64) -> np.ndarray:
    """调用 Ollama embedding 接口生成向量"""
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        resp = ollama.embed(model=model, input=chunk)
        vecs = np.array(resp["embeddings"], dtype=np.float32)
        all_vecs.append(vecs)
    return np.vstack(all_vecs)


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """逐行归一化，保证后续点积等价 cosine 相似度。"""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def build_seed_texts(seed_df: pd.DataFrame, logger=None) -> Tuple[List[str], List[str]]:
    """构造种子库的概念卡片文本与对应 key 列表。"""
    keys = []
    texts = []
    seen = set()
    for idx, r in seed_df.iterrows():
        payload = f"{r.get('类型','')}|{r.get('词名','')}|{r.get('定义','')}"
        if payload in seen:
            if logger:
                logger.log("检测到重复的种子库条目，已用行号进行区分。", "WARNING")
            payload = f"{payload}|row={idx}"
        seen.add(payload)
        keys.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
        texts.append(concept_card_text(r["词名"], r["定义"]))
    return keys, texts


def load_or_build_seed_embeddings(
    seed_df: pd.DataFrame,
    embed_model: str,
    batch_size: int,
    cache_dir: str,
    seed_xlsx: str,
    logger=None
) -> np.ndarray:
    """从缓存加载种子向量，必要时增量补算并更新缓存。"""
    meta_path = os.path.join(cache_dir, "seed_embeddings_meta.json")
    emb_path = os.path.join(cache_dir, "seed_embeddings.npy")
    keys_path = os.path.join(cache_dir, "seed_embeddings_keys.json")

    keys, texts = build_seed_texts(seed_df, logger=logger)
    seed_row_count = len(keys)
    seed_file_hash = _file_sha256(seed_xlsx)
    preprocess_version = preprocess_fingerprint()
    cache_ok = False
    meta = None

    if os.path.exists(meta_path) and os.path.exists(emb_path) and os.path.exists(keys_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json_load(f)
            if meta.get("embed_model") == embed_model and meta.get("preprocess_version") == preprocess_version:
                cache_ok = True
        except Exception:
            cache_ok = False

    if not cache_ok:
        if logger:
            logger.log("Embedding 缓存不可用，执行全量计算...")
        seed_embeddings = embed_texts_ollama(texts, model=embed_model, batch_size=batch_size)
        _save_seed_embedding_cache(
            embed_model, keys, seed_embeddings, seed_row_count, seed_file_hash, preprocess_version, cache_dir
        )
        return seed_embeddings

    try:
        cached_embeddings = np.load(emb_path)
        with open(keys_path, "r", encoding="utf-8") as f:
            cached_keys = json_load(f)
        key_to_vec = {k: cached_embeddings[i] for i, k in enumerate(cached_keys)}
    except Exception:
        if logger:
            logger.log("Embedding 缓存加载失败，执行全量计算...")
        seed_embeddings = embed_texts_ollama(texts, model=embed_model, batch_size=batch_size)
        _save_seed_embedding_cache(
            embed_model, keys, seed_embeddings, seed_row_count, seed_file_hash, preprocess_version, cache_dir
        )
        return seed_embeddings

    if meta:
        prev_rows = meta.get("seed_row_count")
        prev_hash = meta.get("seed_file_hash")
        if (prev_rows is not None and prev_rows != seed_row_count) or (prev_hash and prev_hash != seed_file_hash):
            if logger:
                logger.log("检测到种子库文件发生变更，将按增量方式补算缺失向量并更新缓存。", "WARNING")

    missing_indices = [i for i, k in enumerate(keys) if k not in key_to_vec]
    if missing_indices:
        if logger:
            logger.log(f"检测到 {len(missing_indices)} 条新增/变更概念，进行增量向量计算...")
        missing_texts = [texts[i] for i in missing_indices]
        missing_emb = embed_texts_ollama(missing_texts, model=embed_model, batch_size=batch_size)
        for idx, vec in zip(missing_indices, missing_emb):
            key_to_vec[keys[idx]] = vec

    seed_embeddings = np.vstack([key_to_vec[k] for k in keys])
    _save_seed_embedding_cache(
        embed_model, keys, seed_embeddings, seed_row_count, seed_file_hash, preprocess_version, cache_dir
    )
    return seed_embeddings


def _save_seed_embedding_cache(
    embed_model: str,
    keys: List[str],
    embeddings: np.ndarray,
    seed_row_count: int,
    seed_file_hash: str,
    preprocess_version: str,
    cache_dir: str
) -> None:
    meta_path = os.path.join(cache_dir, "seed_embeddings_meta.json")
    emb_path = os.path.join(cache_dir, "seed_embeddings.npy")
    keys_path = os.path.join(cache_dir, "seed_embeddings_keys.json")

    meta = {
        "embed_model": embed_model,
        "preprocess_version": preprocess_version,
        "seed_row_count": seed_row_count,
        "seed_file_hash": seed_file_hash,
        "count": len(keys)
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json_dump(meta, f)
    with open(keys_path, "w", encoding="utf-8") as f:
        json_dump(keys, f)
    np.save(emb_path, embeddings)


def _file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    """计算文件 SHA256，用于检测种子库变更。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def json_load(f):
    import json
    return json.load(f)


def json_dump(obj, f):
    import json
    json.dump(obj, f, ensure_ascii=False, indent=2)
