"""TinySmoke — load TinyStyler, generate 1 styled + 1 baseline for 1 product."""
import json
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/fs04/ar57/wenyu/.cache/huggingface'
sys.path.insert(0, '/fs04/ar57/wenyu/PersoanlQuery/TinyStyler/tinystyler')

import numpy as np
import torch
from transformers import AutoTokenizer
from tinystyler import TinyStyler

PAPER = '/home/wlia0047/hj82_scratch2/wenyu/e29_paper'

t0 = time.time()
print('A: load TinyStyler', flush=True)
model = TinyStyler(base_model='google/t5-v1_1-large', use_style=True, ctrl_embed_dim=768)
saved = torch.load(
    '/fs04/ar57/wenyu/.cache/huggingface/hub/models--tinystyler--tinystyler/snapshots/2a879107b2ec342e57170b82cdc344d5179fa32b/tinystyler_model_weights.pt',
    map_location='cpu',
)
saved = {k.replace('module.', ''): v for k, v in saved.items()}
cur = model.state_dict()
cur.update(saved)
model.load_state_dict(cur)
model.to('cuda:0').half().eval()
for p in model.parameters():
    p.requires_grad_(False)
print(f'A done {time.time()-t0:.0f}s', flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    '/fs04/ar57/wenyu/.cache/huggingface/hub/models--google--t5-v1_1-large/snapshots/a98b0fcd0b8137ded40cdf0c0cf0ee884e7c9726',
    legacy=True,
)
print(f'B {time.time()-t0:.0f}s', flush=True)

with open(f'{PAPER}/e30_picked_products.json') as f:
    picked = json.load(f)
src = {}
with open(f'{PAPER}/e30_source_queries.jsonl') as f:
    for line in f:
        r = json.loads(line)
        src[r['asin']] = r['source_query']
npz = np.load(f'{PAPER}/e30_style_embs.npz', allow_pickle=True)
uids = {str(u): npz['embs'][i] for i, u in enumerate(npz['user_ids'])}
print(f'C loaded {time.time()-t0:.0f}s', flush=True)

p = picked['products'][0]
users = p['candidate_users'][:2]
sq = src[p['asin']]
print(f'source: {sq[:100]}', flush=True)
texts = [sq] * len(users) + [sq]  # 2 styled + 1 baseline
styles = np.zeros((len(users) + 1, 768), dtype=np.float32)
for i, u in enumerate(users):
    styles[i] = uids[u]
# last row zero (baseline)
enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to('cuda:0')
st = torch.from_numpy(styles).to('cuda:0').half()
print(f'D ready {time.time()-t0:.0f}s ids={enc["input_ids"].shape} style={st.shape}', flush=True)
t1 = time.time()
with torch.no_grad():
    out = model.generate(
        input_ids=enc['input_ids'],
        attention_mask=enc['attention_mask'],
        style=st,
        max_new_tokens=64,
        num_beams=2,
        do_sample=False,
        early_stopping=True,
    )
print(f'GENERATE {time.time()-t1:.0f}s shape={out.shape}', flush=True)
for i, u in enumerate(users):
    txt = tokenizer.decode(out[i], skip_special_tokens=True)
    print(f'  user {u}: {txt[:200]}', flush=True)
print(f'  baseline: {tokenizer.decode(out[-1], skip_special_tokens=True)[:200]}', flush=True)
print(f'TOTAL {time.time()-t0:.0f}s', flush=True)
