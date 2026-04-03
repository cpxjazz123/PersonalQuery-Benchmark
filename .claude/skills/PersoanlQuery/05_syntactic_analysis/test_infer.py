#!/usr/bin/env python3
"""测试模型推理"""
import sys
import torch

# 添加 LingConv 路径
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/LingConv')
from transformers import AutoTokenizer, T5Tokenizer

MODEL_PATH = '/home/wlia0047/ar57_scratch/wenyu/lingconv_checkpoints/0403_21-58-58-ling_disc-t5/best_ling_disc'

print("=" * 60)
print("测试 LINGCONV 模型推理")
print("=" * 60)
print(f"模型路径: {MODEL_PATH}")

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

# 使用官方 EncoderDecoderVAE 类加载模型
from model import EncoderDecoderVAE
from argparse import Namespace

# 创建 args 对象
args = Namespace(
    combine_method='decoder_add_first',
    ling2_only=True,
    ling_embed_type='one-layer',
    lng_dim=40,
    hidden_dim=500,
    disc_lng_dim=40,
    ling_dropout=0.1,
    initializer_range=0.02,
    ling_vae=False,
    sem_loss=False,
    use_semantic_pooling=False,
    pretrain_disc=False,
    disc_loss=False,
    disc_ckpt=None,
    sem_loss_type='shared',
    feedback_param='l',
    max_length=128,
)

pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

model = EncoderDecoderVAE.from_pretrained(MODEL_PATH, args, pad_token_id, eos_token_id)
model = model.to('cuda')
model.eval()

test_inputs = [
    "This is a great product that I would recommend.",
    "The movie was not good at all, very boring.",
    "A person on a horse jumps over a broken down airplane.",
]

print("\n测试结果:")
print("-" * 60)

# 默认的 linguistic features (控制句子复杂度)
# 使用中等复杂度 0.5 作为默认值
default_ling = [0.5] * 40

for i, src_text in enumerate(test_inputs):
    inputs = tokenizer(src_text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    # 添加 linguistic features
    ling_tensor = torch.tensor([default_ling], dtype=torch.float32).to('cuda')

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            sentence1_ling=ling_tensor,
            sentence2_ling=ling_tensor,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[{i+1}] 原文: {src_text}")
    print(f"    生成: {generated}")
    print()

print("=" * 60)
print("测试完成!")