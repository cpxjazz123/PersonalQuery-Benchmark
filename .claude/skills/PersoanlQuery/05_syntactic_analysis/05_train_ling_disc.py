#!/usr/bin/env python3
"""
LingDisc 预训练脚本

功能：
- 训练 LingDisc 判别器，从句子预测 40 维 linguistic features
- 用于 Quality Control (QC) 推理阶段

用法:
    python 05_train_ling_disc.py
"""

import os
import sys
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel, set_seed

# 添加 LingConv 路径
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/LingConv')

from data import LingDataCollator, load_data


class LingDiscClassifier(nn.Module):
    """简化的 LingDisc 分类器 - 用于预训练"""

    def __init__(self, model_name, disc_type="t5", lng_dim=40, hidden_dim=500, dropout=0.1):
        super().__init__()
        self.disc_type = disc_type
        self.lng_dim = lng_dim

        if disc_type == "t5":
            self.encoder = T5EncoderModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.d_model
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, lng_dim)
            )
        else:
            raise ValueError(f"Unknown disc_type: {disc_type}")

    def forward(self, input_ids, attention_mask):
        if self.disc_type == "t5":
            enc_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = enc_output.last_hidden_state.mean(dim=1)  # Mean pooling
            pooled = self.dropout(pooled)
            output = self.fc(pooled)
        return output


class LingDiscTrainer:
    """LingDisc 训练器"""

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        output_dir,
        lr=1e-4,
        weight_decay=0.01,
        epochs=5,
        batch_size=32,
        eval_batch_size=64,
        eval_steps=500,
        max_grad_norm=1.0,
        device=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.eval_steps = eval_steps
        self.max_grad_norm = max_grad_norm

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # 数据整理器
        self.collator = LingDataCollator(tokenizer)

        # 训练状态
        self.global_step = 0
        self.best_eval_loss = float("inf")

    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        if is_best:
            path = os.path.join(self.output_dir, "best_ling_disc")
        else:
            path = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(path, exist_ok=True)

        if self.model.disc_type == "t5":
            self.model.encoder.save_pretrained(path)
        torch.save(self.model.state_dict(), os.path.join(path, "ling_disc.pt"))
        self.tokenizer.save_pretrained(path)

        # 保存配置
        config = {
            "disc_type": self.model.disc_type,
            "lng_dim": self.model.lng_dim,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved checkpoint to {path}")

    def evaluate(self):
        """评估"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                target_ling = batch["sentence2_ling"].to(self.device)

                pred_ling = self.model(input_ids, attention_mask)
                loss = F.mse_loss(pred_ling, target_ling)

                total_loss += loss.item()
                num_batches += 1

        self.model.train()
        return {"eval_loss": total_loss / num_batches}

    def train_step(self, batch):
        """单步训练"""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        target_ling = batch["sentence2_ling"].to(self.device)

        # 前向传播
        pred_ling = self.model(input_ids, attention_mask)
        loss = F.mse_loss(pred_ling, target_ling)

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {"loss": loss.item()}

    def train(self):
        """完整训练"""
        self.model.train()
        print("=" * 70)
        print("LingDisc Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Disc Type: {self.model.disc_type}")
        print(f"Output: {self.output_dir}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch Size: {self.batch_size}")
        print("=" * 70)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_losses = []

            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collator,
            )

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")

            for batch in pbar:
                loss_dict = self.train_step(batch)
                self.global_step += 1

                epoch_losses.append(loss_dict["loss"])

                # 更新进度条
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                # 定期评估
                if self.global_step % self.eval_steps == 0 and self.eval_dataset is not None:
                    eval_metrics = self.evaluate()
                    print(f"\nStep {self.global_step}: eval_loss={eval_metrics['eval_loss']:.4f}")

                    if eval_metrics["eval_loss"] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics["eval_loss"]
                        self.save_checkpoint(is_best=True)
                        print(f"New best eval_loss: {self.best_eval_loss:.4f}")

            # Epoch 结束
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\nEpoch {epoch+1} complete: avg_loss={avg_epoch_loss:.4f}")

            # 每个 epoch 结束后保存
            self.save_checkpoint()

        print("=" * 70)
        print("Training complete!")
        print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/flan-t5-base")
    parser.add_argument("--disc_type", default="t5")
    parser.add_argument("--lng_dim", type=int, default=40)
    parser.add_argument("--hidden_dim", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_dir", default="/home/wlia0047/ar57_scratch/wenyu/lingconv_checkpoints")
    parser.add_argument("--data_dir", default="/home/wlia0047/ar57_scratch/wenyu/ling_conversion_data")
    parser.add_argument("--data", default="ling_conversion")
    args = parser.parse_args()

    # 添加 data_sources 属性
    args.data_sources = ["qqp", "mrpc", "stsb"]
    # 添加缺失的 lng_ids 属性 (None 表示使用所有 40 个特征)
    args.lng_ids = None
    # 添加其他缺失属性
    args.quantize_lng = False
    args.quant_nbins = 20
    args.src_lng = "ling"
    args.do_imputation = False
    args.imputation_percentage = 20
    args.imputation_seed = 0
    args.use_ica = False
    args.n_ica = 10
    args.prepend_prompt = False
    args.prompt_text = "generate a paraphrase: "
    args.use_lingpred = False
    args.aug_same = False
    args.max_eval_samples = 3000
    args.max_length = 128

    # 设置随机种子
    set_seed(args.seed)

    # 构建输出目录
    timestamp = datetime.now().strftime("%m%d_%H-%M-%S")
    output_name = f"{timestamp}-ling_disc-{args.disc_type}"
    output_dir = os.path.join(args.ckpt_dir, output_name)

    print("=" * 70)
    print("LingDisc Training")
    print("=" * 70)
    print(f"Disc Type: {args.disc_type}")
    print(f"Model: {args.model_name}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # 删除旧 checkpoint
    if os.path.exists(output_dir):
        import shutil
        print(f"删除旧 checkpoint: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 加载 tokenizer
    print("\nLoading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    # 加载数据
    print("\nLoading data...")
    data, _, _ = load_data(args, tokenizer, return_data=True)
    print(f"Train: {len(data['train'])} samples")
    if 'dev' in data:
        print(f"Dev: {len(data['dev'])} samples")

    # 保存配置
    config = {
        "disc_type": args.disc_type,
        "model_name": args.model_name,
        "lng_dim": args.lng_dim,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 创建模型
    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LingDiscClassifier(
        model_name=args.model_name,
        disc_type=args.disc_type,
        lng_dim=args.lng_dim,
        hidden_dim=args.hidden_dim,
    )
    model.to(device)

    # 创建训练器
    trainer = LingDiscTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data.get("dev"),
        output_dir=output_dir,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        device=device,
    )

    # 开始训练
    trainer.train()

    # 保存最终模型
    print(f"\nSaving final model to {output_dir}...")
    trainer.save_checkpoint()
    print("Done!")


if __name__ == "__main__":
    main()