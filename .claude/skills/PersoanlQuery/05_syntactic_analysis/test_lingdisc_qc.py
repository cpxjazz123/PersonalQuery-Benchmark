#!/usr/bin/env python3
"""QC 测试 LingDisc 模型"""
import sys
import torch

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/LingConv')
from transformers import T5Tokenizer

MODEL_PATH = '/home/wlia0047/ar57_scratch/wenyu/lingconv_checkpoints/0403_21-58-58-ling_disc-t5/best_ling_disc'

print("=" * 60)
print("QC 测试 LingDisc 模型")
print("=" * 60)
print(f"模型路径: {MODEL_PATH}")

# 加载 tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

# 加载模型
from argparse import Namespace
from transformers import T5EncoderModel
import torch.nn as nn


class LingDiscClassifier(nn.Module):
    """简化的 LingDisc 分类器"""

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
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled = self.dropout(pooled)
        return self.fc(pooled)

args = Namespace(
    model_name="google/flan-t5-base",
    disc_type="t5",
    lng_dim=40,
    hidden_dim=500,
)

model = LingDiscClassifier(
    model_name=args.model_name,
    disc_type=args.disc_type,
    lng_dim=args.lng_dim,
    hidden_dim=args.hidden_dim,
)

# 加载权重
state_dict = torch.load(f"{MODEL_PATH}/ling_disc.pt", map_location='cpu')
model.load_state_dict(state_dict)
model = model.to('cuda')
model.eval()

print(f"模型参数总量: {sum(p.numel() for p in model.parameters()):,}")

test_inputs = [
    "This is a great product that I would recommend to everyone.",
    "The movie was not good at all, very boring and disappointing.",
    "A person on a horse jumps over a broken down airplane.",
    "I absolutely love this item, best purchase ever made!",
    "Terrible experience, would never buy again.",
]

ling_feature_names = [
    "Length", "WordCount", "UniqueWords", "CharCount", "AvgWordLen",
    "SentenceCount", "AvgSentLen", "TypeTokenRatio", "PersonPerspective",
    "HapaxLegomena", "LongWords", "ShortWords", "ComplexWords",
    "SyllablesPerWord", "FleschReadingEase", "FleschKincaidGrade",
    "GunningFogIndex", "SMOGIndex", "ColemanLiauIndex", "ARI",
    "LIX", "RIX", "DaleChallScore", "LinelandScore",
    "DependencyLength", "TreeDepth", "SubordinateClauses",
    "CoordinateClauses", "NPPerVP", "PPPerVP", "AdjPerNP",
    "AdvPerVP", "ModalVerbs", "Pronouns", "DefiniteArticles",
    "IndefiniteArticles", "Prepositions", "Conjunctions", "CoordinatingConj",
    "SubordinatingConj", "AuxiliaryVerbs", "CardinalNumbers"
]

print("\n测试结果:")
print("-" * 60)

for i, src_text in enumerate(test_inputs):
    inputs = tokenizer(src_text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])

    pred_ling = outputs[0].cpu().numpy()

    print(f"[{i+1}] 原文: {src_text}")
    print(f"    预测 linguistic features (前10个):")
    for j in range(min(10, len(ling_feature_names))):
        print(f"      {ling_feature_names[j]}: {pred_ling[j]:.4f}")
    print()

print("=" * 60)
print("QC 测试完成!")
