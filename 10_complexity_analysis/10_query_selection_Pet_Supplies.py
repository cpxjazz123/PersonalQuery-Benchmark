#!/usr/bin/env python3
"""Query selection for Pet_Supplies (load trained model, infer, and select queries)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Pet_Supplies"

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
import train_vades_lite_sentence_latent_threshold as train_module


def log(message: str) -> None:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def main() -> None:
    log(f"Query 选择开始 - CATEGORY: {CATEGORY}")
    
    input_dir = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
    
    candidate_query_file = input_dir / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl"
    sentence_file = input_dir / f"{train_module.OUTPUT_TAG}_sentences.jsonl"
    excluded_user_file = input_dir / f"{train_module.OUTPUT_TAG}_excluded_users.jsonl"
    user_profile_file = input_dir / f"{train_module.OUTPUT_TAG}_user_profiles.jsonl"
    cov_tag = f"_{train_module.COVARIANCE_MODE}" if train_module.COVARIANCE_MODE != "diagonal" else ""
    encoder_path = input_dir / f"vades_encoder_{train_module.OUTPUT_TAG}{cov_tag}.pt"
    user_table_path = input_dir / f"vades_user_table_{train_module.OUTPUT_TAG}{cov_tag}.pt"
    selected_record_file = input_dir / f"{train_module.OUTPUT_TAG}_selected_query_records.jsonl"
    rejected_record_file = input_dir / f"{train_module.OUTPUT_TAG}_rejected_query_records.jsonl"
    query_output_file = REPO_ROOT / "result" / "personal_query" / "06_query" / CATEGORY / f"query_by_expression_style_{train_module.OUTPUT_TAG}.json"
    
    if not encoder_path.exists():
        log(f"错误: 模型不存在: {encoder_path}")
        log(f"请先运行 10_train_only_{CATEGORY}.py 训练模型")
        sys.exit(1)
    
    # Step 1: 加载候选 query
    log("Step 1: 加载候选 query")
    with candidate_query_file.open("r", encoding="utf-8") as f:
        candidate_rows = [json.loads(line) for line in f if line.strip()]
    log(f"候选 query 数: {len(candidate_rows)}")
    candidate_user_ids = {row["user_id"] for row in candidate_rows}
    
    # Step 2: 加载解析结果
    log("Step 2: 加载解析结果")
    with sentence_file.open("r", encoding="utf-8") as f:
        enriched_sentence_rows = [json.loads(line) for line in f if line.strip()]
    enriched_sentence_rows = [row for row in enriched_sentence_rows if row["user_id"] in candidate_user_ids]
    log(f"过滤后解析句子: {len(enriched_sentence_rows)} 条")
    feature_names = list(enriched_sentence_rows[0]["features"].keys())
    
    # Step 3: 加载排除用户
    log("Step 3: 加载排除用户")
    if excluded_user_file.exists():
        with excluded_user_file.open("r", encoding="utf-8") as f:
            excluded_rows = [json.loads(line) for line in f if line.strip()]
    else:
        excluded_rows = []
    log(f"排除用户数: {len(excluded_rows)}")
    
    # Step 4: 构建数据集
    log("Step 4: 构建数据集")
    train_module.REPO_ROOT = REPO_ROOT
    train_module.CATEGORY = CATEGORY
    train_module.INPUT_DIR = input_dir
    user_ids_filtered, dataset = train_module.build_training_dataset(enriched_sentence_rows, feature_names)
    log(f"数据集构建完成: {len(user_ids_filtered)} 用户")
    
    # Step 5: 加载模型
    log("Step 5: 加载模型")
    train_module.DEVICE = train_module.infer_device()
    
    encoder_ckpt = torch.load(encoder_path, map_location=train_module.DEVICE, weights_only=False)
    ckpt_mode = encoder_ckpt.get("covariance_mode", "diagonal")
    if ckpt_mode != train_module.COVARIANCE_MODE:
        raise ValueError(f"covariance_mode 不一致: ckpt={ckpt_mode}, env={train_module.COVARIANCE_MODE}")
    ckpt_encoder_dist = encoder_ckpt.get("encoder_dist", "gaussian")
    if ckpt_encoder_dist != train_module.ENCODER_DIST:
        raise ValueError(f"encoder_dist 不一致: ckpt={ckpt_encoder_dist}, env={train_module.ENCODER_DIST}")
    if train_module.COVARIANCE_MODE == "full":
        encoder = train_module.SentenceEncoderFull(
            input_dim=encoder_ckpt["input_dim"],
            hidden_dim=encoder_ckpt["hidden_dim"],
            latent_dim=encoder_ckpt["latent_dim"]
        ).to(train_module.DEVICE)
    elif train_module.ENCODER_DIST == "student_t":
        encoder = train_module.SentenceEncoderStudentT(
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
    state_dict = encoder_ckpt["model_state_dict"]
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    encoder.load_state_dict(new_state_dict)
    encoder.eval()

    user_table_ckpt = torch.load(user_table_path, map_location=train_module.DEVICE, weights_only=False)
    if train_module.COVARIANCE_MODE == "full":
        user_table = train_module.UserDistributionTableFull(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"]
        ).to(train_module.DEVICE)
    elif train_module.COVARIANCE_MODE == "diagonal_gmm":
        user_table = train_module.UserDistributionTableGMM(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
            num_components=user_table_ckpt.get("gmm_components", int(os.environ.get("VADES_GMM_K", "2"))),
        ).to(train_module.DEVICE)
    elif train_module.COVARIANCE_MODE == "diagonal_student_t":
        user_table = train_module.UserDistributionTableStudentT(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
        ).to(train_module.DEVICE)
    elif train_module.COVARIANCE_MODE == "diagonal_laplace":
        user_table = train_module.UserDistributionTableLaplace(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
        ).to(train_module.DEVICE)
    elif train_module.COVARIANCE_MODE == "diagonal_logistic":
        user_table = train_module.UserDistributionTableLogistic(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
        ).to(train_module.DEVICE)
    elif train_module.COVARIANCE_MODE == "diagonal_student_t_gmm":
        user_table = train_module.UserDistributionTableStudentTGMM(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
            num_components=user_table_ckpt.get("gmm_components", int(os.environ.get("VADES_GMM_K", "2"))),
        ).to(train_module.DEVICE)
    else:
        user_table = train_module.UserDistributionTable(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"]
        ).to(train_module.DEVICE)
    state_dict = user_table_ckpt["model_state_dict"]
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    user_table.load_state_dict(new_state_dict)
    user_table.eval()
    log("模型加载完成")
    
    # Step 6: 推理用户分布
    log("Step 6: 推理用户句子分布")
    sentence_output_rows, user_profile_rows = train_module.infer_user_sentence_distributions(
        encoder, user_table, user_ids_filtered, dataset
    )
    log(f"推理完成: {len(sentence_output_rows)} 句子, {len(user_profile_rows)} 用户")
    
    # Step 7: 过滤用户（只保留有候选 query、user profile 和 holdout 句子的用户）
    log("Step 7: 过滤用户")
    user_profile_user_ids = {row["user_id"] for row in user_profile_rows}
    holdout_indices = np.flatnonzero(dataset["holdout_mask"])
    holdout_user_ids = set()
    for idx in holdout_indices:
        uid = dataset["sentence_rows"][idx]["user_id"]
        holdout_user_ids.add(uid)
    
    # 调试：输出每个条件的数量
    log(f"  - 有候选query的用户: {len(candidate_user_ids)}")
    log(f"  - 有用户画像的用户: {len(user_profile_user_ids)}")
    log(f"  - 通过数据集构建的用户: {len(user_ids_filtered)}")
    log(f"  - holdout集中有句子的用户: {len(holdout_user_ids)}")
    
    # 逐步计算交集
    intermediate_1 = candidate_user_ids & user_profile_user_ids
    log(f"  - candidate & user_profile: {len(intermediate_1)}")
    intermediate_2 = intermediate_1 & set(user_ids_filtered)
    log(f"  - & user_ids_filtered: {len(intermediate_2)}")
    valid_user_ids = intermediate_2 & holdout_user_ids
    log(f"  - & holdout_user_ids: {len(valid_user_ids)}")
    
    valid_user_ids = candidate_user_ids & user_profile_user_ids & set(user_ids_filtered) & holdout_user_ids
    candidate_rows = [row for row in candidate_rows if row["user_id"] in valid_user_ids]
    user_profile_rows = [row for row in user_profile_rows if row["user_id"] in valid_user_ids]
    user_ids_filtered = [uid for uid in user_ids_filtered if uid in valid_user_ids]
    log(f"有效用户数: {len(user_ids_filtered)}")
    
    # Step 8: 校准阈值
    log("Step 8: 校准阈值")
    abs_thresholds = train_module.calibrate_absolute_threshold_with_unseen_holdout(
        encoder, user_table, user_ids_filtered, dataset
    )
    log("阈值校准完成")
    
    # Step 9: 排序和选择 query
    log("Step 9: 排序和选择 query")
    selected_rows, rejected_rows, query_output_rows = train_module.rank_and_select_queries(
        encoder=encoder, user_table=user_table, dataset=dataset,
        candidate_rows=candidate_rows, user_ids=user_ids_filtered,
        user_profile_rows=user_profile_rows, abs_thresholds=abs_thresholds,
    )
    log(f"Query 选择完成: {len(selected_rows)} 选中, {len(rejected_rows)} 拒绝")
    
    # Step 10: 保存结果
    log("Step 10: 保存结果")
    train_module.write_jsonl(sentence_file, sentence_output_rows)
    train_module.write_jsonl(user_profile_file, user_profile_rows)
    train_module.write_jsonl(selected_record_file, selected_rows)
    train_module.write_jsonl(rejected_record_file, rejected_rows)
    query_output_file.parent.mkdir(parents=True, exist_ok=True)
    query_output_file.write_text(json.dumps(query_output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    
    log(f"用户画像已保存: {user_profile_file}")
    log(f"结果已保存: {selected_record_file}")
    log(f"{CATEGORY} Query 选择完成！")


if __name__ == "__main__":
    main()
