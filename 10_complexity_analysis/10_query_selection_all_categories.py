#!/usr/bin/env python3
"""Query selection script for all categories (load trained models and select queries)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
import train_vades_lite_sentence_latent_threshold as train_module


def log(message: str) -> None:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def process_category(category: str) -> None:
    log(f"\n{'='*60}")
    log(f"处理类别: {category}")
    log(f"{'='*60}")
    
    # 设置路径
    input_dir = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / category
    candidate_query_file = input_dir / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl"
    sentence_file = input_dir / f"{train_module.OUTPUT_TAG}_sentences.jsonl"
    excluded_user_file = input_dir / f"{train_module.OUTPUT_TAG}_excluded_users.jsonl"
    user_profile_file = input_dir / f"{train_module.OUTPUT_TAG}_user_profiles.jsonl"
    cov_tag = f"_{train_module.COVARIANCE_MODE}" if train_module.COVARIANCE_MODE != "diagonal" else ""
    encoder_path = input_dir / f"vades_encoder_{train_module.OUTPUT_TAG}{cov_tag}.pt"
    user_table_path = input_dir / f"vades_user_table_{train_module.OUTPUT_TAG}{cov_tag}.pt"
    selected_record_file = input_dir / f"{train_module.OUTPUT_TAG}_selected_query_records.jsonl"
    rejected_record_file = input_dir / f"{train_module.OUTPUT_TAG}_rejected_query_records.jsonl"
    query_output_file = REPO_ROOT / "result" / "personal_query" / "06_query" / category / f"query_by_expression_style_{train_module.OUTPUT_TAG}.json"
    
    # 检查模型文件是否存在
    if not encoder_path.exists():
        log(f"错误: 模型文件不存在: {encoder_path}")
        return
    if not user_table_path.exists():
        log(f"错误: 用户表文件不存在: {user_table_path}")
        return
    
    # Step 1: 加载候选 query
    log("Step 1: 加载候选 query")
    with candidate_query_file.open("r", encoding="utf-8") as f:
        candidate_rows = [json.loads(line) for line in f if line.strip()]
    log(f"候选 query 数: {len(candidate_rows)}")
    
    # Step 2: 加载解析结果
    log("Step 2: 加载解析结果")
    with sentence_file.open("r", encoding="utf-8") as f:
        enriched_sentence_rows = [json.loads(line) for line in f if line.strip()]
    log(f"已加载 {len(enriched_sentence_rows)} 条解析句子")
    
    # 获取 feature_names
    feature_names = list(enriched_sentence_rows[0]["features"].keys())
    
    # Step 3: 加载排除用户
    log("Step 3: 加载排除用户")
    if excluded_user_file.exists():
        with excluded_user_file.open("r", encoding="utf-8") as f:
            excluded_rows = [json.loads(line) for line in f if line.strip()]
    else:
        excluded_rows = []
    log(f"排除用户数: {len(excluded_rows)}")
    
    # 获取用户 ID 列表
    user_ids = list({row["user_id"] for row in enriched_sentence_rows})
    log(f"用户数: {len(user_ids)}")
    
    # Step 4: 构建数据集
    log("Step 4: 构建数据集")
    train_module.REPO_ROOT = REPO_ROOT
    train_module.CATEGORY = category
    train_module.INPUT_DIR = input_dir
    user_ids_filtered, dataset = train_module.build_training_dataset(enriched_sentence_rows, feature_names)
    log(f"数据集构建完成: {len(user_ids_filtered)} 用户")
    
    # Step 5: 加载模型
    log("Step 5: 加载训练好的模型")
    train_module.DEVICE = train_module.infer_device()
    
    # 加载 encoder
    encoder_ckpt = torch.load(encoder_path, map_location=train_module.DEVICE, weights_only=False)
    ckpt_mode = encoder_ckpt.get("covariance_mode", "diagonal")
    if ckpt_mode != train_module.COVARIANCE_MODE:
        raise ValueError(f"covariance_mode 不一致: ckpt={ckpt_mode}, env={train_module.COVARIANCE_MODE}")
    if train_module.COVARIANCE_MODE == "full":
        encoder = train_module.SentenceEncoderFull(
            input_dim=encoder_ckpt["input_dim"],
            hidden_dim=encoder_ckpt["hidden_dim"],
            latent_dim=encoder_ckpt["latent_dim"]
        ).to(train_module.DEVICE)
    else:
        encoder = train_module.SentenceEncoder(
            input_dim=encoder_ckpt["input_dim"],
            hidden_dim=encoder_ckpt["hidden_dim"],
            latent_dim=encoder_ckpt["latent_dim"]
        ).to(train_module.DEVICE)
    encoder.load_state_dict(encoder_ckpt["model_state_dict"])
    encoder.eval()

    # 加载 user_table
    user_table_ckpt = torch.load(user_table_path, map_location=train_module.DEVICE, weights_only=False)
    if train_module.COVARIANCE_MODE == "full":
        user_table = train_module.UserDistributionTableFull(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"]
        ).to(train_module.DEVICE)
    else:
        user_table = train_module.UserDistributionTable(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"]
        ).to(train_module.DEVICE)
    user_table.load_state_dict(user_table_ckpt["model_state_dict"])
    user_table.eval()
    
    log("模型加载完成")
    
    # Step 6: 加载用户画像
    log("Step 6: 加载用户画像")
    with user_profile_file.open("r", encoding="utf-8") as f:
        user_profile_rows = [json.loads(line) for line in f if line.strip()]
    log(f"用户画像数: {len(user_profile_rows)}")
    
    # Step 7: 校准阈值
    log("Step 7: 校准阈值")
    abs_thresholds = train_module.calibrate_absolute_threshold_with_unseen_holdout(
        encoder, user_table, user_ids_filtered, dataset
    )
    log("阈值校准完成")
    
    # Step 8: 排序和选择 query
    log("Step 8: 排序和选择 query")
    selected_rows, rejected_rows, query_output_rows = train_module.rank_and_select_queries(
        encoder=encoder,
        user_table=user_table,
        dataset=dataset,
        candidate_rows=candidate_rows,
        user_ids=user_ids_filtered,
        user_profile_rows=user_profile_rows,
        abs_thresholds=abs_thresholds,
    )
    log(f"Query 选择完成: {len(selected_rows)} 选中, {len(rejected_rows)} 拒绝")
    
    # Step 9: 保存结果
    log("Step 9: 保存结果")
    train_module.write_jsonl(selected_record_file, selected_rows)
    train_module.write_jsonl(rejected_record_file, rejected_rows)
    query_output_file.parent.mkdir(parents=True, exist_ok=True)
    query_output_file.write_text(json.dumps(query_output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    
    log(f"结果已保存:")
    log(f"  - 选中 query: {selected_record_file}")
    log(f"  - 拒绝 query: {rejected_record_file}")
    log(f"  - Query 输出: {query_output_file}")
    log(f"类别 {category} 处理完成！")


def main() -> None:
    log("开始 Query 选择流程 (所有类别)")
    log(f"类别列表: {CATEGORIES}")
    
    for category in CATEGORIES:
        try:
            process_category(category)
        except Exception as e:
            log(f"处理类别 {category} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    log("\n所有类别处理完成！")


if __name__ == "__main__":
    main()
