#!/usr/bin/env python3
"""Run train-only pipeline for Baby_Products (load cached parse results and train model)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Baby_Products"

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
import train_vades_lite_sentence_latent_threshold as train_module


def log(message: str) -> None:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def main() -> None:
    train_module.REPO_ROOT = REPO_ROOT
    train_module.CATEGORY = CATEGORY
    train_module.INPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
    train_module.REVIEW_SOURCE_FILE = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / CATEGORY / "stage1_filtered_users_reviews.json"
    train_module.RAW_CANDIDATE_QUERY_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / CATEGORY / "query_by_expression_style_no_depth_check_10.json"
    train_module.CANDIDATE_QUERY_FILE = train_module.INPUT_DIR / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl"
    train_module.SENTENCE_EXTRACT_CACHE_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_extracted_sentences.jsonl"
    train_module.SUMMARY_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_summary.json"
    train_module.DETAIL_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_epoch_details.jsonl"
    train_module.USER_PROFILE_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_user_profiles.jsonl"
    train_module.SENTENCE_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_sentences.jsonl"
    train_module.EXCLUDED_USER_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_excluded_users.jsonl"
    train_module.SELECTED_RECORD_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_selected_query_records.jsonl"
    train_module.REJECTED_RECORD_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_rejected_query_records.jsonl"
    train_module.QUERY_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / CATEGORY / f"query_by_expression_style_{train_module.OUTPUT_TAG}.json"

    log(f"训练阶段开始 - CATEGORY: {CATEGORY}")
    
    # 检查解析结果是否存在
    if not train_module.SENTENCE_FILE.exists():
        log(f"错误: 解析结果不存在，请先运行解析脚本")
        log(f"期望文件: {train_module.SENTENCE_FILE}")
        sys.exit(1)
    
    # Step 1: 加载解析结果
    log("Step 1: 加载解析结果")
    with train_module.SENTENCE_FILE.open("r", encoding="utf-8") as f:
        enriched_sentence_rows = [json.loads(line) for line in f if line.strip()]
    log(f"已加载 {len(enriched_sentence_rows)} 条解析句子")
    
    # 获取 feature_names
    if enriched_sentence_rows:
        feature_names = list(enriched_sentence_rows[0]["features"].keys())
    else:
        log("错误: 没有解析句子")
        sys.exit(1)
    
    # Step 2: 加载排除用户
    log("Step 2: 加载排除用户")
    if train_module.EXCLUDED_USER_FILE.exists():
        with train_module.EXCLUDED_USER_FILE.open("r", encoding="utf-8") as f:
            excluded_rows = [json.loads(line) for line in f if line.strip()]
    else:
        excluded_rows = []
    log(f"排除用户数: {len(excluded_rows)}")
    
    # Step 3: 构建训练数据集
    log("Step 3: 构建训练数据集")
    user_ids_filtered, dataset = train_module.build_training_dataset(enriched_sentence_rows, feature_names)
    log(f"训练数据集构建完成: {len(user_ids_filtered)} 用户")
    
    # Step 4: 训练模型
    log("Step 4: 训练 VADES 模型")
    train_module.DEVICE = train_module.infer_device()
    train_module.set_random_seed(train_module.SEED)
    log(f"运行设备: {train_module.DEVICE}")
    encoder, user_table, training_info = train_module.train_vades_user_distribution_model(user_ids_filtered, dataset, feature_names)
    log("模型训练完成")
    
    # Step 5: 保存模型
    log("Step 5: 保存模型")
    cov_tag = f"_{train_module.COVARIANCE_MODE}" if train_module.COVARIANCE_MODE != "diagonal" else ""
    encoder_path = train_module.INPUT_DIR / f"vades_encoder_{train_module.OUTPUT_TAG}{cov_tag}.pt"
    user_table_path = train_module.INPUT_DIR / f"vades_user_table_{train_module.OUTPUT_TAG}{cov_tag}.pt"
    torch = sys.modules.get("torch")
    if torch is None:
        import torch
    input_dim = len(feature_names)
    torch.save({"model_state_dict": encoder.state_dict(), "input_dim": input_dim, "hidden_dim": train_module.HIDDEN_DIM, "latent_dim": train_module.LATENT_DIM, "covariance_mode": train_module.COVARIANCE_MODE, "encoder_dist": train_module.ENCODER_DIST}, encoder_path)
    user_table_save = {
        "model_state_dict": user_table.state_dict(),
        "num_users": len(user_ids_filtered),
        "latent_dim": train_module.LATENT_DIM,
        "covariance_mode": train_module.COVARIANCE_MODE,
    }
    if train_module.COVARIANCE_MODE in {"diagonal_gmm", "diagonal_student_t_gmm"}:
        user_table_save["gmm_components"] = train_module.GMM_COMPONENTS
    torch.save(user_table_save, user_table_path)
    log(f"模型已保存: {encoder_path}, {user_table_path}")
    
    # Step 6: 保存训练摘要
    log("Step 6: 保存训练摘要")
    summary = {
        "category": CATEGORY,
        "num_users": len(user_ids_filtered),
        "num_sentences": len(enriched_sentence_rows),
        "num_excluded_users": len(excluded_rows),
        "feature_names": feature_names,
        "training_epochs": training_info["epochs"],
        "model_files": {
            "encoder": str(encoder_path),
            "user_table": str(user_table_path),
        }
    }
    train_module.SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    train_module.write_jsonl(train_module.DETAIL_FILE, training_info["epochs"])
    
    log("训练阶段完成！")
    log(f"模型文件:")
    log(f"  - Encoder: {encoder_path}")
    log(f"  - User Table: {user_table_path}")
    log(f"  - 摘要: {train_module.SUMMARY_FILE}")
    log(f"下一步: 运行 10_query_selection_{CATEGORY}.py 进行推理和查询选择")


if __name__ == "__main__":
    main()
