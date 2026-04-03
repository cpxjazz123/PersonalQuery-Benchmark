#!/usr/bin/env python3
"""
LINGCONV 完整训练流程

训练步骤:
1. 训练 LingDisc (语言判别器) - 从句子预测 40 维语言学特征
2. 训练主模型 (EncoderDecoderVAE) - 带 LingDisc 和 SemEmb 辅助损失

用法:
    python 05_train_full.py
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
from model import EncoderDecoderVAE


# ========================================
# LingDisc 分类器
# ========================================
class LingDiscClassifier(nn.Module):
    """LingDisc 分类器 - 从句子预测 40 维语言学特征"""

    def __init__(self, model_name, lng_dim=40, hidden_dim=500, dropout=0.1):
        super().__init__()
        self.lng_dim = lng_dim
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.d_model
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, lng_dim)
        )

    def forward(self, input_ids, attention_mask):
        enc_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = enc_output.last_hidden_state.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


# ========================================
# 语义损失计算器
# ========================================
class SemanticLossComputer:
    """计算语义相似度损失"""

    def __init__(self, model, device, loss_weight=0.1):
        self.model = model
        self.device = device
        self.loss_weight = loss_weight

    def compute(self, input_ids, attention_mask, labels):
        """计算源句和目标句的语义相似度损失"""
        try:
            # 编码源句
            source_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            source_emb = source_outputs.last_hidden_state.mean(1)

            # 编码目标句
            if labels is not None:
                decoder_input_ids = self.model._shift_right(labels)
            else:
                return None, 0

            with torch.no_grad():
                decoder_embeds = self.model.decoder.embed_tokens(decoder_input_ids)
                decoder_outputs = self.model.decoder(
                    inputs_embeds=decoder_embeds,
                    encoder_hidden_states=source_outputs.last_hidden_state,
                    encoder_attention_mask=attention_mask,
                )
                target_emb = decoder_outputs.last_hidden_state.mean(1)

            # 余弦相似度
            cos_sim = F.cosine_similarity(source_emb, target_emb, dim=-1)
            sem_loss = (1 - cos_sim).mean()

            return sem_loss, self.loss_weight * sem_loss

        except Exception as e:
            print(f"Warning: Semantic loss failed: {e}")
            return None, 0


# ========================================
# 完整训练器
# ========================================
class FullTrainer:
    """完整训练器 - 支持 LingDisc + SemEmb 辅助损失"""

    def __init__(
        self,
        model,
        ling_disc,
        tokenizer,
        train_dataset,
        eval_dataset,
        output_dir,
        ling_disc_ckpt=None,
        sem_loss_weight=0.1,
        disc_loss_weight=0.1,
        lr=1e-4,
        weight_decay=1e-2,
        epochs=2,
        batch_size=16,
        eval_batch_size=32,
        eval_steps=500,
        max_grad_norm=1.0,
        device=None,
    ):
        self.model = model
        self.ling_disc = ling_disc
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.ling_disc_ckpt = ling_disc_ckpt
        self.sem_loss_weight = sem_loss_weight
        self.disc_loss_weight = disc_loss_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.eval_steps = eval_steps
        self.max_grad_norm = max_grad_norm

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.ling_disc:
            self.ling_disc.to(self.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # 数据整理器
        self.collator = LingDataCollator(tokenizer)

        # 语义损失计算器
        self.sem_computer = SemanticLossComputer(model, self.device, sem_loss_weight)

        # 训练状态
        self.global_step = 0
        self.best_eval_loss = float("inf")

    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        if is_best:
            path = os.path.join(self.output_dir, "best_model")
        else:
            path = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Saved checkpoint to {path}")

    def evaluate(self):
        """评估"""
        self.model.eval()
        total_loss = 0
        total_lm_loss = 0
        total_disc_loss = 0
        total_sem_loss = 0
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
                labels = batch["labels"].to(self.device)
                sentence2_ling = batch.get("sentence2_ling")
                if sentence2_ling is not None:
                    sentence2_ling = sentence2_ling.to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    sentence2_ling=sentence2_ling,
                )

                lm_loss = outputs.loss

                # Disc 损失
                disc_loss = 0
                if self.ling_disc and self.disc_loss_weight > 0:
                    pred_ling = self.ling_disc(input_ids, attention_mask)
                    disc_loss = F.mse_loss(pred_ling, sentence2_ling)

                # Sem 损失
                sem_loss_val = 0
                if self.sem_loss_weight > 0:
                    sem_loss, weighted_sem = self.sem_computer.compute(input_ids, attention_mask, labels)
                    if sem_loss is not None:
                        sem_loss_val = sem_loss.item()

                total_loss_item = lm_loss.item() + self.disc_loss_weight * disc_loss + self.sem_loss_weight * sem_loss_val
                total_loss += total_loss_item
                total_lm_loss += lm_loss.item()
                total_disc_loss += disc_loss.item() if disc_loss else 0
                total_sem_loss += sem_loss_val
                num_batches += 1

        self.model.train()
        return {
            "eval_loss": total_loss / num_batches,
            "lm_loss": total_lm_loss / num_batches,
            "disc_loss": total_disc_loss / num_batches,
            "sem_loss": total_sem_loss / num_batches,
        }

    def train_step(self, batch):
        """单步训练"""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        sentence2_ling = batch.get("sentence2_ling")
        if sentence2_ling is not None:
            sentence2_ling = sentence2_ling.to(self.device)

        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            sentence2_ling=sentence2_ling,
        )

        lm_loss = outputs.loss

        # Disc 损失
        disc_loss = 0
        if self.ling_disc and self.disc_loss_weight > 0:
            pred_ling = self.ling_disc(input_ids, attention_mask)
            disc_loss = F.mse_loss(pred_ling, sentence2_ling)

        # Sem 损失
        sem_loss_value = 0
        sem_loss_weighted = 0
        if self.sem_loss_weight > 0:
            sem_loss, weighted_sem = self.sem_computer.compute(input_ids, attention_mask, labels)
            if sem_loss is not None:
                sem_loss_value = sem_loss.item()
                sem_loss_weighted = weighted_sem

        # 总损失
        total_loss = lm_loss + self.disc_loss_weight * disc_loss + sem_loss_weighted

        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "lm_loss": lm_loss.item(),
            "disc_loss": disc_loss.item() if disc_loss else 0,
            "sem_loss": sem_loss_value,
            "total_loss": total_loss.item(),
        }

    def train(self):
        """完整训练"""
        self.model.train()
        print("=" * 70)
        print("LINGCONV Full Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"LingDisc: {self.ling_disc is not None}")
        print(f"Sem Loss Weight: {self.sem_loss_weight}")
        print(f"Disc Loss Weight: {self.disc_loss_weight}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Output: {self.output_dir}")
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

                epoch_losses.append(loss_dict["total_loss"])

                # 更新进度条
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lm": f"{loss_dict['lm_loss']:.4f}",
                    "disc": f"{loss_dict['disc_loss']:.4f}",
                    "sem": f"{loss_dict['sem_loss']:.4f}",
                })

                # 定期评估
                if self.global_step % self.eval_steps == 0 and self.eval_dataset is not None:
                    eval_metrics = self.evaluate()
                    print(f"\nStep {self.global_step}: {eval_metrics}")

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
    # 模型参数
    parser.add_argument("--model_name", default="google/flan-t5-base")
    parser.add_argument("--combine_method", default="decoder_add_first")
    parser.add_argument("--ling2_only", type=bool, default=True)
    parser.add_argument("--ling_embed_type", default="one-layer")
    parser.add_argument("--lng_dim", type=int, default=40)
    parser.add_argument("--hidden_dim", type=int, default=500)
    parser.add_argument("--disc_lng_dim", type=int, default=40)
    parser.add_argument("--ling_dropout", type=float, default=0.1)
    parser.add_argument("--initializer_range", type=float, default=0.02)

    # 训练参数
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # 损失权重
    parser.add_argument("--sem_loss_weight", type=float, default=0.1)
    parser.add_argument("--disc_loss_weight", type=float, default=0.1)
    parser.add_argument("--ling_vae", type=bool, default=False)

    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--ckpt_dir", default="/home/wlia0047/ar57_scratch/wenyu/lingconv_checkpoints")
    parser.add_argument("--ling_disc_ckpt", default=None, help="LingDisc checkpoint path")

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

    # 设置随机种子
    set_seed(args.seed)

    # 构建输出目录
    timestamp = datetime.now().strftime("%m%d_%H-%M-%S")
    output_name = f"{timestamp}-ling_conversion-full"
    output_dir = os.path.join(args.ckpt_dir, output_name)

    print("=" * 70)
    print("LINGCONV Full Training")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Sem Loss Weight: {args.sem_loss_weight}")
    print(f"Disc Loss Weight: {args.disc_loss_weight}")
    print(f"LingDisc Checkpoint: {args.ling_disc_ckpt}")
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
    print("Loading data...")
    data, _, _ = load_data(args, tokenizer, return_data=True)
    print(f"Train: {len(data['train'])} samples")
    if 'dev' in data:
        print(f"Dev: {len(data['dev'])} samples")

    # 加载 LingDisc
    ling_disc = None
    if args.ling_disc_ckpt and os.path.exists(args.ling_disc_ckpt):
        print(f"\nLoading LingDisc from {args.ling_disc_ckpt}...")
        ling_disc = LingDiscClassifier(
            model_name=args.model_name,
            lng_dim=args.lng_dim,
            hidden_dim=args.hidden_dim,
        )
        ling_disc.load_state_dict(torch.load(os.path.join(args.ling_disc_ckpt, "ling_disc.pt"), map_location='cpu'))
        ling_disc.eval()
    else:
        print("\nWarning: No LingDisc checkpoint provided, disc_loss will be 0")

    # 加载主模型
    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用 get_model
    from model import get_model
    model, _, _ = get_model(args, tokenizer, device)
    model.train()

    # 保存配置
    config = {
        "model_name": args.model_name,
        "combine_method": args.combine_method,
        "ling2_only": args.ling2_only,
        "sem_loss_weight": args.sem_loss_weight,
        "disc_loss_weight": args.disc_loss_weight,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "ling_disc_ckpt": args.ling_disc_ckpt,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 创建训练器
    trainer = FullTrainer(
        model=model,
        ling_disc=ling_disc,
        tokenizer=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data.get("dev"),
        output_dir=output_dir,
        ling_disc_ckpt=args.ling_disc_ckpt,
        sem_loss_weight=args.sem_loss_weight,
        disc_loss_weight=args.disc_loss_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_steps=args.eval_steps,
        max_grad_norm=args.max_grad_norm,
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