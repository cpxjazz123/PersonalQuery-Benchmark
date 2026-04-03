#!/usr/bin/env python3
"""
LINGCONV 统一脚本
===================
支持训练、测试、推理等功能

使用方法:
    python lingconv.py --mode train --combine_method decoder_add_first
    python lingconv.py --mode test --ckpt <checkpoint_path>
    python lingconv.py --mode infer --ckpt <checkpoint_path>
"""

import argparse
import json
import os
import sys
import types
from copy import deepcopy
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.linear_model import Ridge
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.pipeline import Pipeline
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainerCallback,
)


# ========================================
# 常量定义
# ========================================
lca_names = [
    "WBL", "CBL", "TBL", "ABL", "VLBL", "CND", "CTND", "TAN", "MST", "MGT",
    "MLT", "STT", "CPT", "CST", "CGT", "MST", "SRL", "PRD", "ADP", "PCJ",
    "LGS", "CLS", "FCD", "COD", "SOD", "LCS", "CCS", "DCS", "PDC", "ITJ",
    "PTJ", "OPN", "CPN", "ANM", "CNM", "NNM", "NPM", "AJM", "VNM", "RLM"
]

lingfeat_names = [
    "F1_1", "F1_2", "F1_3", "F1_4", "F1_5", "F1_6", "F1_7", "F1_8",
    "F2_1", "F2_2", "F2_3", "F2_4", "F2_5", "F2_6", "F2_7", "F2_8",
    "F3_1", "F3_2", "F3_3", "F3_4", "F3_5", "F3_6", "F3_7", "F3_8",
    "F4_1", "F4_2", "F4_3", "F4_4", "F4_5", "F4_6", "F4_7", "F4_8"
]

sca_names = [
    "SRL_1", "SRL_2", "SRL_3", "SRL_4", "SRL_5", "SRL_6", "SRL_7", "SRL_8",
    "PRD_1", "PRD_2", "PRD_3", "PRD_4", "PRD_5", "PRD_6", "PRD_7", "PRD_8",
    "ADP_1", "ADP_2", "ADP_3", "ADP_4", "ADP_5", "ADP_6", "ADP_7", "ADP_8",
    "PCJ_1", "PCJ_2", "PCJ_3", "PCJ_4", "PCJ_5", "PCJ_6", "PCJ_7", "PCJ_8"
]

used_indices = list(range(40))


# ========================================
# 参数解析
# ========================================
def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"yes", "true", "t", "y", "1"}:
        return True
    if lowered in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="LINGCONV - Linguistically-controlled Paraphrase Generation")
    parser.add_argument("--mode", default="train", choices=["train", "test", "infer", "evaluate"],
                        help="运行模式: train(训练), test(测试), infer(推理), evaluate(评估)")

    # ========== 硬编码参数 ==========
    args = parser.parse_args()
    args.mode = "train"  # 硬编码为训练模式
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_predict", action="store_true")

    # 数据参数
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--data", default="ling_conversion")
    parser.add_argument("--src_lng", default="ling")
    parser.add_argument("--split", default="test")

    # 训练参数
    parser.add_argument("--ckpt_dir", default="./checkpoints")
    parser.add_argument("--ckpt")
    parser.add_argument("--predict_fn", default="preds/predictions.txt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--grad_accumulation", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)

    # 模型参数
    parser.add_argument("--model_name", default="google/flan-t5-base")
    parser.add_argument("--combine_method", default="decoder_add_first",
                        choices=["decoder_add", "decoder_add_first", "decoder_concat",
                                "bos_replace", "layer_injection", "input_concat", "input_add"])
    parser.add_argument("--ling2_only", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--ling_embed_type", default="one-layer", choices=["one-layer", "two-layer"])
    parser.add_argument("--injection_type", default="first")
    parser.add_argument("--injection_layer", type=int, default=1)
    parser.add_argument("--combine_weight", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=500)
    parser.add_argument("--lng_dim", type=int, default=40)
    parser.add_argument("--disc_lng_dim", type=int, default=40)
    parser.add_argument("--ling_dropout", type=float, default=0.1)
    parser.add_argument("--initializer_range", type=float, default=0.02)

    # 精度参数
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    # 其他参数
    parser.add_argument("--use_semantic_pooling", action="store_true")
    parser.add_argument("--sem_loss", action="store_true")
    parser.add_argument("--disc_loss", action="store_true")
    parser.add_argument("--pretrain_disc", action="store_true")
    parser.add_argument("--pretrain_sem", action="store_true")
    parser.add_argument("--pretrain_gen", action="store_true")
    parser.add_argument("--ling_vae", action="store_true")
    parser.add_argument("--linggen_type", default="none")

    # 推理参数
    parser.add_argument("--input_text", help="推理时的输入文本")
    parser.add_argument("--complexity", type=float, default=0.5, help="复杂度向量 (0-1)")

    args = parser.parse_args()

    # ========== 硬编码参数 ==========
    args.do_train = True
    args.data = "ling_conversion"
    args.ckpt_dir = "/home/wlia0047/ar57_scratch/wenyu/lingconv_checkpoints"
    args.model_name = "google/flan-t5-base"
    args.batch_size = 64  # per_device_train_batch_size
    args.grad_accumulation = 4  # 有效批次大小 = 64 * 4 = 256
    args.epochs = 2
    args.lr = 1e-4  # T5 默认学习率，降低以减少训练不稳定
    args.max_length = 64  # 减小序列长度，加速训练
    args.combine_method = "decoder_add_first"
    args.bf16 = True
    args.gradient_checkpointing = False  # 关闭，用时间换显存不适合我们
    args.warmup_steps = 1000
    args.weight_decay = 1e-2
    args.max_grad_norm = 10.0
    args.seed = 0
    args.lng_dim = 40
    args.hidden_dim = 500
    args.disc_lng_dim = 40
    args.ling_dropout = 0.1
    args.initializer_range = 0.02
    args.ling_embed_type = "one-layer"
    args.ling2_only = True
    args.use_semantic_pooling = False
    args.sem_loss = False
    args.disc_loss = False
    args.pretrain_disc = False
    args.pretrain_sem = False
    args.pretrain_gen = False
    args.ling_vae = False
    args.linggen_type = "none"
    args.early_stopping_patience = 3

    # 生成输出目录名
    args.name = f"{datetime.now().strftime('%m%d_%H-%M-%S')}-{args.data}-{args.combine_method}"

    return args


# ========================================
# GPU监控回调
# ========================================
class GPUMonitorCallback(TrainerCallback):
    def __init__(self, logging_steps=50):
        self.logging_steps = logging_steps

    def on_step_end(self, args, state, control, **kwargs):
        import torch
        if state.global_step % self.logging_steps == 0 and state.global_step > 0:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"[Step {state.global_step}] GPU显存: 已用 {allocated:.2f} GB / 预留 {reserved:.2f} GB / 总计 {total:.2f} GB")


# ========================================
# 数据处理
# ========================================
def load_data(args, tokenizer):
    """加载ling_conversion数据集，支持缓存"""
    data_dir = "/home/wlia0047/ar57_scratch/wenyu/ling_conversion_data"
    cache_dir = os.path.join(os.path.dirname(data_dir), "ling_conversion_data_cache")

    # 检查缓存是否存在
    if os.path.exists(cache_dir):
        print(f"发现缓存，加载缓存数据: {cache_dir}")
        data = load_from_disk(cache_dir)
        return data, None, None

    # 无缓存，加载官方数据集 (qqp,mrpc,stsb only)
    print(f"未发现缓存，从 HuggingFace 加载官方数据集 (qqp,mrpc,stsb)...")
    from datasets import concatenate_datasets, load_dataset

    def filter_positive(example):
        if "label" in example:
            return example["label"] in [0, 1] and example["label"] >= 1 if "stsb" not in example.get("source", "") else example["label"] >= 3
        return True

    def normalize_columns(data, source_name):
        rename_map = {}
        if "question1" in data.column_names and "question2" in data.column_names:
            rename_map["question1"] = "sentence1"
            rename_map["question2"] = "sentence2"
        if rename_map:
            data = data.rename_columns(rename_map)
        if "source" not in data.column_names:
            data = data.add_column("source", [source_name] * len(data))
        keep_cols = [c for c in data.column_names if c.startswith("sentence") or c == "source"]
        return data.remove_columns([c for c in data.column_names if c not in keep_cols])

    # 加载各个数据集
    datasets = []
    source_specs = {
        "qqp": {"keep": lambda x: x["label"] == 1},
        "mrpc": {"keep": lambda x: x["label"] == 1},
        "stsb": {"keep": lambda x: x["label"] >= 3.0},
    }

    for name, spec in source_specs.items():
        print(f"  加载 {name}...")
        ds = load_dataset("glue", name, split="train")
        ds = ds.filter(spec["keep"])
        ds = normalize_columns(ds, name)
        datasets.append(ds)

    data = concatenate_datasets(datasets)
    print(f"  官方数据集样本数: {len(data)}")

    # 分割成 train/dev/test
    from datasets import DatasetDict
    train_dev = data.train_test_split(test_size=0.1, seed=42)
    dev_test = train_dev["test"].train_test_split(test_size=0.5, seed=42)
    data = DatasetDict({
        "train": train_dev["train"],
        "dev": dev_test["train"],
        "test": dev_test["test"],
    })
    print(f"  分割后: train={len(data['train'])}, dev={len(data['dev'])}, test={len(data['test'])}")

    def preprocess_function(examples):
        inputs = examples["sentence1"]
        targets = examples["sentence2"]

        model_inputs = tokenizer(inputs, max_length=args.max_length,
                                truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=args.max_length,
                          truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    for split in data.keys():
        # 添加 linguistic features 占位符
        if "sentence1_ling" not in data[split].column_names:
            data[split] = data[split].add_column("sentence1_ling", [[0.0]*40]*len(data[split]))
        if "sentence2_ling" not in data[split].column_names:
            data[split] = data[split].add_column("sentence2_ling", [[0.0]*40]*len(data[split]))

        data[split] = data[split].map(
            preprocess_function,
            batched=True,
            remove_columns=[c for c in data[split].column_names if c not in ["sentence1_ling", "sentence2_ling", "input_ids", "attention_mask", "labels"]],
            desc=f"Tokenizing {split}"
        )

    # 保存缓存
    print(f"保存缓存到: {cache_dir}")
    data.save_to_disk(cache_dir)

    return data, None, None


class LingDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        batch = super().__call__(features)
        return batch


# ========================================
# 模型定义 - 使用官方 LingConv 实现
# ========================================
# 导入官方实现
lingconv_base_path = "/home/wlia0047/ar57/wenyu/LingConv"
sys.path.insert(0, lingconv_base_path)
from lingconv_t5 import LingConvT5ForConditionalGeneration
from model import EncoderDecoderVAE


def get_model(args, tokenizer, device):
    """创建LINGCONV模型 - 使用官方EncoderDecoderVAE类"""
    from transformers import T5Config

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    if args.ckpt and os.path.exists(args.ckpt):
        # 从checkpoint加载
        print(f"从checkpoint加载模型: {args.ckpt}")
        model = EncoderDecoderVAE.from_pretrained(args.ckpt, args, pad_token_id, eos_token_id).to(device)
    else:
        # 从头创建
        print(f"创建新模型: {args.model_name}")
        config = T5Config.from_pretrained(args.model_name)
        model = EncoderDecoderVAE.from_pretrained(args.model_name, args, pad_token_id, eos_token_id).to(device)

    return model, model.config, None


# ========================================
# 训练函数
# ========================================
def train(args):
    """训练模型"""
    import torch
    from transformers import set_seed

    print("=" * 60)
    print("GPU DEBUG INFO")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  总显存: {props.total_memory / 1024**3:.2f} GB")
        print(f"当前GPU: {torch.cuda.current_device()}")
    else:
        print("CUDA不可用!")
    print("=" * 60)

    set_seed(args.seed)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    data, _, _ = load_data(args, tokenizer)

    output_dir = os.path.join(args.ckpt_dir, args.name) if args.name else args.ckpt_dir

    # 删除旧的 checkpoint 目录
    if os.path.exists(output_dir):
        import shutil
        print(f"删除旧 checkpoint: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    has_eval = "dev" in data

    # 日志打印批次大小信息
    effective_batch_size = args.batch_size * args.grad_accumulation
    print(f"[BATCH INFO] per_device_train_batch_size: {args.batch_size}")
    print(f"[BATCH INFO] gradient_accumulation_steps: {args.grad_accumulation}")
    print(f"[BATCH INFO] effective_batch_size: {effective_batch_size}")
    print(f"[BATCH INFO] gradient_checkpointing: {args.gradient_checkpointing}")
    print(f"[BATCH INFO] num_train_epochs: {args.epochs}")
    print(f"[BATCH INFO] total_training_steps: {19730 * args.epochs / args.grad_accumulation}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=50,
        evaluation_strategy="steps" if args.do_train and has_eval else "no",
        eval_steps=500 if args.do_train and has_eval else None,
        save_strategy="steps" if args.do_train and has_eval else ("epoch" if args.do_train else "no"),
        save_steps=500 if args.do_train and has_eval else None,
        save_total_limit=2,
        load_best_model_at_end=bool(args.do_train and has_eval),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        fp16=args.fp16 if hasattr(args, 'fp16') else False,
        bf16=args.bf16 if hasattr(args, 'bf16') else False,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    model, _, _ = get_model(args, tokenizer, training_args.device)
    collator = LingDataCollator(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        train_dataset=data.get("train"),
        eval_dataset=data.get("dev"),
    )

    trainer.add_callback(GPUMonitorCallback(logging_steps=50))

    if args.do_train:
        print("开始训练...")
        trainer.train(resume_from_checkpoint=args.ckpt if args.ckpt else None)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"训练完成! 模型保存于: {output_dir}")

    if args.do_eval:
        if "dev" not in data:
            raise ValueError("No dev split found for evaluation.")
        dev_metrics = trainer.evaluate(eval_dataset=data["dev"])
        print(dev_metrics)

    if args.do_predict:
        if args.split not in data:
            raise ValueError(f"Split {args.split!r} not found in dataset.")
        predictions = trainer.predict(data[args.split]).predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        output_path = os.path.join(output_dir, args.predict_fn)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        decoded = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        with open(output_path, "w") as f:
            f.write("\n".join(decoded))
        print(f"预测结果保存于: {output_path}")


# ========================================
# 测试函数
# ========================================
def test(args):
    """测试模型"""
    import torch
    from transformers import set_seed

    if not args.ckpt:
        ckpt_dirs = [d for d in os.listdir(args.ckpt_dir) if os.path.isdir(os.path.join(args.ckpt_dir, d))]
        if ckpt_dirs:
            latest = sorted(ckpt_dirs)[-1]
            args.ckpt = os.path.join(args.ckpt_dir, latest)
            checkpoints = [d for d in os.listdir(args.ckpt) if d.startswith("checkpoint-")]
            if checkpoints:
                args.ckpt = os.path.join(args.ckpt, sorted(checkpoints)[-1])

    if not args.ckpt or not os.path.exists(args.ckpt):
        print(f"错误: Checkpoint路径不存在: {args.ckpt}")
        return

    print(f"使用Checkpoint: {args.ckpt}")

    tokenizer = T5Tokenizer.from_pretrained(args.ckpt)
    model, _, _ = get_model(args, tokenizer, torch.device('cuda:0'))
    model.eval()

    data, _, _ = load_data(args, tokenizer)

    print("=" * 60)
    print("测试结果")
    print("=" * 60)

    test_split = data.get("test", data.get("dev"))
    if test_split is None:
        print("错误: 没有找到测试集")
        return

    for i in range(min(10, len(test_split))):
        sample = test_split[i]
        src = sample.get('sentence1', None)
        tgt = sample.get('sentence2', None)
        ling = sample.get('sentence2_ling', [0.5]*40)

        if src is None:
            continue

        inputs = tokenizer(src, return_tensors='pt', truncation=True, max_length=128)
        ling_tensor = torch.tensor([ling], dtype=torch.float32)

        batch = {
            "input_ids": inputs['input_ids'].cuda(),
            "attention_mask": inputs['attention_mask'].cuda(),
            "sentence1_ling": ling_tensor.cuda(),
            "sentence2_ling": ling_tensor.cuda(),
            "labels": inputs['input_ids'].cuda(),
        }

        with torch.no_grad():
            pred = model.infer(batch)

        generated = tokenizer.decode(pred[0], skip_special_tokens=True)

        print(f"\n[样本 {i+1}]")
        print(f"输入:   {src}")
        print(f"目标:   {tgt}")
        print(f"生成:   {generated}")

    print("\n" + "=" * 60)


# ========================================
# 推理函数
# ========================================
def infer(args):
    """交互式推理"""
    import torch

    if not args.ckpt:
        ckpt_dirs = [d for d in os.listdir(args.ckpt_dir) if os.path.isdir(os.path.join(args.ckpt_dir, d))]
        if ckpt_dirs:
            latest = sorted(ckpt_dirs)[-1]
            args.ckpt = os.path.join(args.ckpt_dir, latest)
            checkpoints = [d for d in os.listdir(args.ckpt) if d.startswith("checkpoint-")]
            if checkpoints:
                args.ckpt = os.path.join(args.ckpt, sorted(checkpoints)[-1])

    if not args.ckpt or not os.path.exists(args.ckpt):
        print(f"错误: Checkpoint路径不存在: {args.ckpt}")
        return

    print(f"使用Checkpoint: {args.ckpt}")

    tokenizer = T5Tokenizer.from_pretrained(args.ckpt)
    model, _, _ = get_model(args, tokenizer, torch.device('cuda:0'))
    model.eval()

    complexity = max(0.0, min(1.0, args.complexity))
    ling = [complexity] * 40

    print("\n输入句子进行测试 (输入quit退出):")
    while True:
        try:
            line = input("> ")
            if line.lower() in ['quit', 'q', 'exit']:
                break

            inputs = tokenizer(line, return_tensors='pt', truncation=True, max_length=128)
            ling_tensor = torch.tensor([ling], dtype=torch.float32)

            batch = {
                "input_ids": inputs['input_ids'].cuda(),
                "attention_mask": inputs['attention_mask'].cuda(),
                "sentence1_ling": ling_tensor,
                "sentence2_ling": ling_tensor,
                "labels": inputs['input_ids'].cuda(),
            }

            with torch.no_grad():
                pred = model.infer(batch)

            generated = tokenizer.decode(pred[0], skip_special_tokens=True)
            print(f"生成: {generated}")
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n退出")
            break


# ========================================
# 主入口
# ========================================
if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    elif args.mode == "infer":
        infer(args)
    elif args.mode == "evaluate":
        args.do_eval = True
        train(args)
    else:
        print(f"未知模式: {args.mode}")
        print("可用模式: train, test, infer, evaluate")
        sys.exit(1)
