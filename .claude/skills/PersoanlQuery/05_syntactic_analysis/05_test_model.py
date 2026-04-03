#!/usr/bin/env python3
"""测试训练好的 LINGCONV 模型"""

import torch
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

# 模型路径
MODEL_PATH = "/home/wlia0047/ar57_scratch/wenyu/lingconv_checkpoints/0403_20-01-34-ling_conversion-decoder_add_first"

def main():
    print("=" * 70)
    print("测试 LINGCONV 模型生成")
    print("=" * 70)
    print(f"模型路径: {MODEL_PATH}")

    # 加载 tokenizer 和模型
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"设备: {device}")
    print("-" * 70)

    # 测试句子
    test_inputs = [
        "A person on a horse jumps over a broken down airplane.",
        "This is a great product that I would recommend to anyone.",
        "The movie was not good at all, very boring.",
    ]

    print("\n测试句子:")
    print("-" * 70)

    for i, src_text in enumerate(test_inputs):
        # 编码
        inputs = tokenizer(src_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )

        # 解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\n[{i+1}] 原文: {src_text}")
        print(f"    生成: {generated_text}")

    print("\n" + "=" * 70)
    print("测试完成!")

if __name__ == "__main__":
    main()