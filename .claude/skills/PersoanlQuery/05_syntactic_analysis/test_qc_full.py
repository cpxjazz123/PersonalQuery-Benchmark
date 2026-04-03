#!/usr/bin/env python3
"""QC 测试：主模型 + LingDisc + SemEmb 进行完整质量控制"""

import sys
import torch
import torch.nn.functional as F
import logging

# 配置调试日志
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/LingConv')
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, T5EncoderModel
from argparse import Namespace

# 模型路径
MAIN_MODEL_PATH = '/home/wlia0047/ar57_scratch/wenyu/lingconv_checkpoints/0403_20-22-16-ling_conversion-decoder_add_first'
LINGDISC_PATH = '/home/wlia0047/ar57_scratch/wenyu/lingconv_checkpoints/0403_21-58-58-ling_disc-t5/best_ling_disc'

logger.info("=" * 70)
logger.info("QC 测试：主模型 + LingDisc + SemEmb")
logger.info("=" * 70)
logger.info(f"主模型路径: {MAIN_MODEL_PATH}")
logger.info(f"LingDisc路径: {LINGDISC_PATH}")

# ============ 加载 tokenizer ============
logger.info("开始加载 tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(MAIN_MODEL_PATH)
logger.info(f"tokenizer vocab size: {tokenizer.vocab_size}")
logger.info(f"pad_token_id: {tokenizer.pad_token_id}, eos_token_id: {tokenizer.eos_token_id}")

# ============ 定义 LingDisc 模型 ============
class LingDiscClassifier(torch.nn.Module):
    def __init__(self, model_name, disc_type="t5", lng_dim=40, hidden_dim=500, dropout=0.1):
        super().__init__()
        self.disc_type = disc_type
        self.lng_dim = lng_dim

        if disc_type == "t5":
            self.encoder = T5EncoderModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.d_model
            self.dropout = torch.nn.Dropout(dropout)
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, lng_dim)
            )
        else:
            raise ValueError(f"Unknown disc_type: {disc_type}")

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.fc(pooled)

# ============ 加载 LingDisc ============
print("\n加载 LingDisc 模型...")
ling_disc_args = Namespace(
    model_name="google/flan-t5-base",
    disc_type="t5",
    lng_dim=40,
    hidden_dim=500,
)
ling_disc = LingDiscClassifier(
    model_name=ling_disc_args.model_name,
    disc_type=ling_disc_args.disc_type,
    lng_dim=ling_disc_args.lng_dim,
    hidden_dim=ling_disc_args.hidden_dim,
)
state_dict = torch.load(f"{LINGDISC_PATH}/ling_disc.pt", map_location='cpu', weights_only=False)
ling_disc.load_state_dict(state_dict)
print(f"  LingDisc 参数总量: {sum(p.numel() for p in ling_disc.parameters()):,}")

# ============ 加载主模型 ============
print("\n加载主模型 (EncoderDecoderVAE)...")
import os
os.chdir('/home/wlia0047/ar57/wenyu/LingConv')
from model import EncoderDecoderVAE

main_args = Namespace(
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
    sem_loss_type='shared',  # 使用共享 encoder 作为 sem_emb
    feedback_param='l',
    max_length=128,
)

pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

main_model = EncoderDecoderVAE.from_pretrained(MAIN_MODEL_PATH, main_args, pad_token_id, eos_token_id)
print(f"  主模型参数总量: {sum(p.numel() for p in main_model.parameters()):,}")

# ============ SemEmb 使用共享 encoder ============
sem_emb = main_model.encoder  # 共享 encoder 作为 sem_emb

# ============ 移动到 GPU ============
device = torch.device('cuda')
ling_disc = ling_disc.to(device)
main_model = main_model.to(device)
ling_disc.eval()
main_model.eval()

print(f"\n设备: {device}")

# ============ 测试用例 ============
logger.info("开始准备测试用例...")
test_cases = [
    {
        "src_text": "This is a great product that I would recommend to everyone.",
        "target_ling": [0.5] * 40,  # 目标写作风格（中等复杂度）
    },
    {
        "src_text": "I absolutely love this item, best purchase ever made!",
        "target_ling": [0.3] * 40,  # 目标写作风格（较简单）
    },
    {
        "src_text": "The movie was not good at all, very boring and disappointing.",
        "target_ling": [0.7] * 40,  # 目标写作风格（较复杂）
    },
]
logger.info(f"共加载 {len(test_cases)} 个测试用例")

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

print("\n" + "=" * 70)
print("QC 测试结果")
print("=" * 70)
logger.info("测试开始 - 最终验证")

for i, case in enumerate(test_cases):
    src_text = case["src_text"]
    target_ling = case["target_ling"]

    print(f"\n[案例 {i+1}]")
    print(f"  原文: {src_text}")
    print(f"  目标风格: {target_ling[:5]}... (前5维)")

    # 编码输入
    logger.debug(f"[案例 {i+1}] 开始编码输入...")
    inputs = tokenizer(src_text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logger.debug(f"  input_ids shape: {inputs['input_ids'].shape}")
    logger.debug(f"  attention_mask shape: {inputs['attention_mask'].shape}")

    # 构建 batch
    ling_tensor = torch.tensor([target_ling], dtype=torch.float32).to(device)
    logger.debug(f"  ling_tensor shape: {ling_tensor.shape}")
    batch = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'sentence2_ling': ling_tensor,
        'sentence1_ling': ling_tensor,
    }
    logger.debug(f"[案例 {i+1}] Batch 构建完成")

    # ============ 测试 infer_with_feedback_BP ============
    print(f"\n  [QC with Feedback BP]")
    logger.info(f"[案例 {i+1}] 开始推理...")
    with torch.no_grad():
        # 初步生成 - infer_with_cache 返回 (output, cache)
        logger.debug(f"[案例 {i+1}] 调用 main_model.infer_with_cache...")
        logger.debug(f"[案例 {i+1}] batch keys: {list(batch.keys())}")
        logger.debug(f"[案例 {i+1}] batch input_ids shape: {batch['input_ids'].shape}")
        logger.debug(f"[案例 {i+1}] batch sentence1_ling shape: {batch['sentence1_ling'].shape}")
        logger.debug(f"[案例 {i+1}] batch sentence2_ling shape: {batch['sentence2_ling'].shape}")
        dec_output, cache = main_model.infer_with_cache(batch)
        logger.debug(f"[案例 {i+1}] infer_with_cache 返回完成")
        # dec_output 是生成模型的输出，cache 包含中间结果
        logger.debug(f"[案例 {i+1}] cache keys: {list(cache.keys())}")
        for k, v in cache.items():
            if isinstance(v, torch.Tensor):
                logger.debug(f"[案例 {i+1}] cache['{k}'] shape: {v.shape}, dtype: {v.dtype}")
            elif isinstance(v, (list, tuple)):
                logger.debug(f"[案例 {i+1}] cache['{k}'] type: {type(v)}, len: {len(v)}")
            else:
                logger.debug(f"[案例 {i+1}] cache['{k}'] type: {type(v)}, value: {v}")
        logits = cache.get('logits')
        pred_ids = dec_output  # dec_output 直接就是生成的 token ids
        logger.debug(f"[案例 {i+1}] pred_ids shape: {pred_ids.shape}")
        pred_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        logger.info(f"[案例 {i+1}] 生成文本: {pred_text}")
        print(f"    初步生成: {pred_text} TEST")

        # 获取 logits 进行 LingDisc 评估
        logits = cache.get('logits')
        if logits is not None:
            logger.debug(f"[案例 {i+1}] logits shape: {logits.shape}")
            logger.debug(f"[案例 {i+1}] logits[:, -1:, :] shape: {logits[:, -1:, :].shape}")
            logits_reshaped = logits[:, -1:, :].reshape(1, -1, logits.shape[-1])
            logger.debug(f"[案例 {i+1}] logits_reshaped for ling_disc: {logits_reshaped.shape}")
            ling_pred = ling_disc(logits_reshaped)
            logger.debug(f"[案例 {i+1}] ling_pred shape: {ling_pred.shape}")
            ling_pred_values = ling_pred[0].cpu().numpy()
            mse_loss = F.mse_loss(ling_pred, ling_tensor).item()
            logger.info(f"[案例 {i+1}] LingDisc MSE Loss: {mse_loss:.4f}")
            print(f"    LingDisc MSE Loss: {mse_loss:.4f}")
            print(f"    预测风格 (前5维): {ling_pred_values[:5]}")

            # 语义相似度（使用共享 encoder）
            logger.debug(f"[案例 {i+1}] 计算语义相似度...")
            s1_emb = main_model.encoder(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            ).last_hidden_state.mean(1)
            logger.debug(f"  s1_emb shape: {s1_emb.shape}")

            s2_ids = tokenizer(pred_text, return_tensors='pt', padding=True, truncation=True, max_length=128)['input_ids'].to(device)
            s2_emb = main_model.encoder(input_ids=s2_ids).last_hidden_state.mean(1)
            logger.debug(f"  s2_emb shape: {s2_emb.shape}")
            cos_sim = F.cosine_similarity(s1_emb, s2_emb, dim=-1).item()
            logger.info(f"[案例 {i+1}] 语义相似度 (cosine): {cos_sim:.4f}")
            print(f"    语义相似度 (cosine): {cos_sim:.4f}")
        else:
            logger.warning(f"[案例 {i+1}] logits 为 None，无法进行 LingDisc 评估")

    print()

print("=" * 70)
print("QC 测试完成!")
