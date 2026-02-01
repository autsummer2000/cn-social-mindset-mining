import os
import math
import numpy as np
import pandas as pd
import ollama
import matplotlib.pyplot as plt

# ========= 配置区 =========
SEED_XLSX = r"D:\博二\AI for science discovery\种子库\种子库_8.xlsx"
SEED_SHEET = "Sheet1"

EMBED_MODELS = [
    "qwen3-embedding:8b",
]

GENERATED_CONCEPT = {
    "academic_label": "算法情绪共振",
    "core_definition": "由平台算法通过优先推送高情绪唤醒内容所引发并维持的、在用户群体中快速扩散的协同性情绪状态。"
}

TOP_K = 20
BATCH_SIZE = 64

OUTPUT_PREFIX = "生成概念_相似度结果"
FIG_DIR = "figs"                 # 图输出目录
MAX_FULL_MATRIX_N = 2500         # 超过这个规模，不建议构造完整 NxN 相似度矩阵
PAIR_SAMPLE_SIZE = 200000        # 任意两概念相似度的随机抽样数量（用于分布可视化）


def concept_card_text(term, definition):
    term = (term or "").strip()
    definition = (definition or "").strip() if definition else ""
    if definition:
        return "概念：{}。定义：{}".format(term, definition)
    return "概念：{}。".format(term)


def load_seed_library(xlsx_path, sheet):
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    required_cols = set(["类型", "词名", "定义"])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError("种子库缺少列：{}；当前列：{}".format(missing, list(df.columns)))

    df["类型"] = df["类型"].astype(str).str.strip()
    df["词名"] = df["词名"].astype(str).str.strip()
    df["定义"] = df["定义"].astype(str).str.strip()
    df = df[df["词名"].notna() & (df["词名"] != "")]
    return df.reset_index(drop=True)


def embed_texts_ollama(texts, model, batch_size=64):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        resp = ollama.embed(model=model, input=chunk)
        vecs = np.array(resp["embeddings"], dtype=np.float32)
        all_vecs.append(vecs)
    return np.vstack(all_vecs)


def normalize_rows(x, eps=1e-12):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def safe_filename_from_model(model_name):
    return model_name.replace(":", "_").replace("/", "_").replace("-", "_")


def seed_nn_stats(seed_emb):
    """
    计算每个 seed 的最近邻最大相似度 nn_max、以及最小相似度 nn_min。
    若 N 不大，则直接算 sim 矩阵；若 N 很大，可改成分块/近邻检索（这里先给直接版）。
    """
    sim = seed_emb.dot(seed_emb.T)
    np.fill_diagonal(sim, -np.inf)
    nn_max = sim.max(axis=1)

    # 为了算最小值，需要把对角线设为 +inf 再取 min
    np.fill_diagonal(sim, np.inf)
    nn_min = sim.min(axis=1)

    # 全局（排除对角线）最小/最大
    # 全局最大：nn_max 的 max
    global_max = float(nn_max.max())

    # 全局最小：nn_min 的 min
    global_min = float(nn_min.min())

    return nn_max, nn_min, global_min, global_max


def sample_pairwise_sims(seed_emb, sample_size=200000, rng_seed=42):
    """
    随机抽样任意两概念相似度（用于估计“整体相关性背景分布”），避免取出全量 off-diagonal。
    """
    rng = np.random.default_rng(rng_seed)
    n = seed_emb.shape[0]
    if n < 2:
        return np.array([], dtype=np.float32)

    # 随机选 i,j，且避免 i==j
    i = rng.integers(0, n, size=sample_size, endpoint=False)
    j = rng.integers(0, n, size=sample_size, endpoint=False)
    mask = (i != j)
    i = i[mask]
    j = j[mask]
    sims = np.sum(seed_emb[i] * seed_emb[j], axis=1)  # dot since normalized
    return sims.astype(np.float32)


def pca_2d(X):
    """
    简单 PCA 2D（不依赖 sklearn），用于可视化嵌入空间。
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    # 协方差矩阵
    C = np.dot(Xc.T, Xc) / max(1, (Xc.shape[0] - 1))
    # 特征分解
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)[::-1]
    W = vecs[:, idx[:2]]
    Z = np.dot(Xc, W)
    return Z


def plot_hist(data, title, xlabel, out_png, bins=60):
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_pca(seed_df, Z, out_png, annotate_top=25):
    """
    画 PCA 2D 散点；为避免过密，只标注少量点（可按需要改成标注离群/高 nn_max 的点）。
    """
    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], s=8)
    plt.title("Seed library embedding PCA (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # 标注前 annotate_top 个点（你也可以改成标注 nn_max 最大的点）
    m = min(annotate_top, Z.shape[0])
    for k in range(m):
        plt.text(Z[k, 0], Z[k, 1], str(seed_df.loc[k, "词名"]), fontsize=7)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def run_one_model(seed_df, seed_texts, gen_text, embed_model):
    os.makedirs(FIG_DIR, exist_ok=True)

    print("\n" + "=" * 80)
    print("Running model:", embed_model)
    print("Seed size:", len(seed_texts))
    print("=" * 80)

    seed_emb = embed_texts_ollama(seed_texts, model=embed_model, batch_size=BATCH_SIZE)
    gen_emb = embed_texts_ollama([gen_text], model=embed_model, batch_size=1)[0]
    seed_emb = normalize_rows(seed_emb)
    gen_emb = normalize_rows(gen_emb.reshape(1, -1))[0]

    # 生成概念 vs 种子库
    sims = seed_emb.dot(gen_emb)
    result_df = seed_df.copy()
    result_df["similarity"] = sims
    result_df = result_df.sort_values("similarity", ascending=False).reset_index(drop=True)

    # 种子库内部统计
    n = seed_emb.shape[0]
    if n <= MAX_FULL_MATRIX_N:
        nn_max, nn_min, global_min, global_max = seed_nn_stats(seed_emb)
    else:
        # 大规模时：至少先算 nn_max/nn_min 需要 sim 矩阵，会很重；
        # 这里给出保守策略：不算 nn_min/global_min（可改成 FAISS 近邻 + 抽样）
        print("[WARN] Seed size too large for full NxN sim; only sampling pairwise distribution.")
        nn_max = np.array([], dtype=np.float32)
        nn_min = np.array([], dtype=np.float32)
        global_min, global_max = float("nan"), float("nan")

    # 抽样任意两概念分布
    pair_sims = sample_pairwise_sims(seed_emb, sample_size=PAIR_SAMPLE_SIZE)

    # 阈值建议：基于分布分位数（见下文解释）
    # 1) U：避免“复刻”——用 nn_max 的高分位（如果 nn_max 为空，则退化用 pair_sims 的高分位）
    if nn_max.size > 0:
        U_95 = float(np.quantile(nn_max, 0.95))
        U_90 = float(np.quantile(nn_max, 0.90))
    else:
        U_95 = float(np.quantile(pair_sims, 0.995))
        U_90 = float(np.quantile(pair_sims, 0.99))

    # 2) L：避免“过于不相似”——建议用 pair_sims 的较高分位（代表“至少与某类概念有中等相关”）
    L_75 = float(np.quantile(pair_sims, 0.75))
    L_80 = float(np.quantile(pair_sims, 0.80))

    gen_max = float(result_df.loc[0, "similarity"])

    # 两套判定：宽松 / 严格
    band_lenient = (L_75, U_95)
    band_strict  = (L_80, U_90)

    def verdict_for_band(L, U, x):
        if x < L:
            return "过于不相似（建议补强定义/边界/指标）"
        if x > U:
            return "过于相似（高重叠风险：疑似改写/近义）"
        return "通过（进入下一步）"

    verdict_lenient = verdict_for_band(band_lenient[0], band_lenient[1], gen_max)
    verdict_strict  = verdict_for_band(band_strict[0], band_strict[1], gen_max)

    print("\n--- Generated concept vs seed ---")
    print("Top1 similarity (max): {:.4f}".format(gen_max))
    print("Lenient band  [L75, U95] = [{:.4f}, {:.4f}] => {}".format(band_lenient[0], band_lenient[1], verdict_lenient))
    print("Strict  band  [L80, U90] = [{:.4f}, {:.4f}] => {}".format(band_strict[0], band_strict[1], verdict_strict))

    print("\n--- Top Similar Seed Concepts (Top {}) ---".format(TOP_K))
    print(result_df.head(TOP_K)[["类型", "词名", "similarity"]])

    # 导出 Excel
    out_base = "{}_{}".format(OUTPUT_PREFIX, safe_filename_from_model(embed_model))
    out_xlsx = out_base + ".xlsx"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="generated_vs_seed", index=False)

        # 种子内部统计（若可得）
        if nn_max.size > 0:
            seed_internal = seed_df.copy()
            seed_internal["nn_max"] = nn_max
            seed_internal["nn_min"] = nn_min
            seed_internal = seed_internal.sort_values("nn_max", ascending=False).reset_index(drop=True)
            seed_internal.to_excel(writer, sheet_name="seed_nn_stats", index=False)

            summary = pd.DataFrame([{
                "embed_model": embed_model,
                "seed_size": n,
                "global_min_offdiag": global_min,
                "global_max_offdiag": global_max,
                "nn_max_q90": float(np.quantile(nn_max, 0.90)),
                "nn_max_q95": float(np.quantile(nn_max, 0.95)),
                "pair_q75": L_75,
                "pair_q80": L_80,
                "band_lenient_L": band_lenient[0],
                "band_lenient_U": band_lenient[1],
                "band_strict_L": band_strict[0],
                "band_strict_U": band_strict[1],
                "generated_max_similarity": gen_max,
                "verdict_lenient": verdict_lenient,
                "verdict_strict": verdict_strict,
                "generated_concept_text": gen_text
            }])
        else:
            summary = pd.DataFrame([{
                "embed_model": embed_model,
                "seed_size": n,
                "pair_q75": L_75,
                "pair_q80": L_80,
                "band_lenient_L": band_lenient[0],
                "band_lenient_U": band_lenient[1],
                "band_strict_L": band_strict[0],
                "band_strict_U": band_strict[1],
                "generated_max_similarity": gen_max,
                "verdict_lenient": verdict_lenient,
                "verdict_strict": verdict_strict,
                "generated_concept_text": gen_text
            }])
        summary.to_excel(writer, sheet_name="run_summary", index=False)

        # 抽样分布
        pd.DataFrame({"pairwise_sim_sample": pair_sims}).to_excel(writer, sheet_name="pairwise_sample", index=False)

    print("Saved:", out_xlsx)

    # 画图：nn_max / pairwise sample / PCA
    if nn_max.size > 0:
        plot_hist(
            nn_max,
            title="Seed NN max similarity distribution ({})".format(embed_model),
            xlabel="nn_max (each concept's max similarity to others)",
            out_png=os.path.join(FIG_DIR, out_base + "_nn_max_hist.png"),
            bins=60
        )

    plot_hist(
        pair_sims,
        title="Sampled pairwise similarity distribution ({})".format(embed_model),
        xlabel="cosine similarity (sampled off-diagonal pairs)",
        out_png=os.path.join(FIG_DIR, out_base + "_pairwise_hist.png"),
        bins=80
    )

    # PCA 可视化（点很多时会密，但能看整体结构）
    Z = pca_2d(seed_emb)
    plot_pca(seed_df, Z, os.path.join(FIG_DIR, out_base + "_pca.png"), annotate_top=25)


def main():
    seed_df = load_seed_library(SEED_XLSX, SEED_SHEET)
    seed_texts = [concept_card_text(r["词名"], r["定义"]) for _, r in seed_df.iterrows()]
    gen_text = concept_card_text(GENERATED_CONCEPT.get("academic_label", ""), GENERATED_CONCEPT.get("core_definition", ""))

    print("\n=== Generated Concept ===")
    print(gen_text)

    for embed_model in EMBED_MODELS:
        try:
            run_one_model(seed_df, seed_texts, gen_text, embed_model)
        except Exception as e:
            print("\n[ERROR] Model failed:", embed_model)
            print("Reason:", repr(e))
            print("建议先确认：ollama list 里存在该模型；必要时执行 ollama pull {}".format(embed_model))


if __name__ == "__main__":
    main()
