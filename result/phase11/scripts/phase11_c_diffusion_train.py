#!/usr/bin/env python3
"""Phase 11.C: Conditional Latent Diffusion training.

Trains DDPM denoiser epsilon_theta(z_t, t, c, s_u) where:
  z_t  : noisy latent (LATENT_DIM=128)
  t    : diffusion timestep
  c    : content embedding from Qwen2-7B layer 14, mean-pooled, PCA->128d
  s_u  : user style sample ~ N(mu_u, Sigma_u)

Architecture: MLP with FiLM conditioning (4 ResBlocks, hidden=512).

Steps:
  1. Extract Qwen hidden states for ~12000 phase10 train sentences (43770 if full)
  2. PCA 3584 -> 128 on content embeddings
  3. Compute per-user Sigma_red in 128d: V_k @ Sigma @ V_k^T
  4. For each (sentence, user) pair, sample s_u and train denoiser
  5. Save denoiser.pt + meta + training curve

Output:
  phase11_c_diffusion_unet.pt
  phase11_c_diffusion_meta.json
  phase11_c_content_embs_128d.npy + phase11_c_content_pca.npy
  phase11_c_training_curve.png
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
RAW_SENTENCES_FILE = VADES_DIR / "vades_prototype_3000u_v6_raw_sentences.jsonl"

PHASE10_PAIRS = OUT_DIR / "phase10_pairs_1000.jsonl"
GAUSSIANS_NPZ = OUT_DIR / "phase11_a_user_gaussians_768d.npz"

# === Output ===
OUT_DENOISER = OUT_DIR / "phase11_c_diffusion_unet.pt"
OUT_META = OUT_DIR / "phase11_c_diffusion_meta.json"
OUT_CONTENT_EMB = OUT_DIR / "phase11_c_content_embs_128d.npy"
OUT_CONTENT_PCA = OUT_DIR / "phase11_c_content_pca.npz"
OUT_CURVE_PNG = OUT_DIR / "phase11_c_training_curve.png"

# === Hardcoded config ===
LATENT_DIM = 128
CONTENT_DIM = 3584     # Qwen2-7B hidden dim
HIDDEN_DIFF_DIM = 512
N_DIFF_BLOCKS = 4
N_TIMESTEPS = 1000
BETA_SCHEDULE = "cosine"
BATCH_SIZE = 512        # ↑ from 256 (model is small)
N_EPOCHS = 30           # ↑ from 15 (each epoch ~0.25s with AMP+BATCH512)
LR = 1e-4               # middle ground (was 2e-4 unstable, 5e-5 too slow)
EMA_DECAY = 0.9999
QWEN_LAYER_FOR_CONTENT = 14
QWEN_HIDDEN_BATCH = 64   # ↑ from 32 (Qwen extraction faster)
QWEN_HIDDEN_MAX_LEN = 96
RANDOM_SEED = 42
USE_AMP = True           # mixed precision (bfloat16) for 2-3x speedup (after film fix)
PRE_SAMPLE_S_U = True    # pre-sample all s_u before training (avoid per-batch sampling)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# === Diffusion components ===

def make_beta_schedule(n_timesteps: int, schedule: str = "cosine") -> np.ndarray:
    """Cosine beta schedule (improved DDPM)."""
    if schedule == "cosine":
        s = 0.008
        x = np.linspace(0, n_timesteps, n_timesteps + 1)
        alphas_cumprod = np.cos(((x / n_timesteps) + s) / (1 + s) * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 1e-4, 0.999)
    elif schedule == "linear":
        return np.linspace(1e-4, 0.02, n_timesteps)
    else:
        raise ValueError(f"unknown schedule {schedule}")


class GaussianDiffusion:
    """Minimal DDPM helper."""

    def __init__(self, n_timesteps: int, schedule: str = "cosine", device: str = "cpu"):
        self.n_timesteps = n_timesteps
        self.betas = make_beta_schedule(n_timesteps, schedule)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = torch.from_numpy(np.sqrt(self.alphas_cumprod).astype(np.float32)).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.from_numpy(np.sqrt(1.0 - self.alphas_cumprod).astype(np.float32)).to(device)
        self.alphas_cumprod_torch = torch.from_numpy(self.alphas_cumprod.astype(np.float32)).to(device)
        self.device = device

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: q(z_t | z_0)."""
        sa = self.sqrt_alphas_cumprod[t][:, None]  # [B, 1]
        so = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return sa * z0 + so * noise

    def sample_timesteps(self, n: int, device: str) -> torch.Tensor:
        return torch.randint(0, self.n_timesteps, (n,), device=device)


# === Sinusoidal time embedding (standard) ===

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        x = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(F.silu(self.fc1(self.ln(x))))
        return x + h


def film_modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class ConditionalDenoiser(nn.Module):
    """epsilon_theta(z_t, t, c, s_u) -> predicted noise.

    NOTE: c is already PCA-projected (latent_dim), not raw 3584d.
    """

    def __init__(self, latent_dim=LATENT_DIM, content_dim=LATENT_DIM, style_dim=LATENT_DIM,
                 hidden_dim=HIDDEN_DIFF_DIM, n_blocks=N_DIFF_BLOCKS):
        super().__init__()
        self.style_proj = nn.Linear(style_dim, hidden_dim)
        self.content_proj = nn.Linear(content_dim, hidden_dim)
        self.t_embed = SinusoidalPosEmb(dim=hidden_dim)
        self.in_proj = nn.Linear(latent_dim + 3 * hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.film = nn.ModuleList([nn.Linear(hidden_dim, 2 * hidden_dim) for _ in range(n_blocks)])
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, c_raw: torch.Tensor, s_u: torch.Tensor) -> torch.Tensor:
        """
        Args:
          z_t: [B, latent_dim]
          t: [B] timesteps
          c_raw: [B, latent_dim] PCA-projected content embedding
          s_u: [B, latent_dim] user style sample (already in PCA space)
        """
        t_e = self.t_embed(t)
        c_e = self.content_proj(c_raw)
        s_e = self.style_proj(s_u)
        cond = t_e + c_e + s_e
        h = self.in_proj(torch.cat([z_t, t_e, c_e, s_e], dim=-1))
        for blk, film in zip(self.blocks, self.film):
            scale, shift = film(cond).chunk(2, dim=-1)
            h = blk(h)
            h = film_modulate(h, scale, shift)
        return self.out_proj(h)


# === EMA helper ===

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)


# === Dataset ===

class StyleContentDataset(Dataset):
    def __init__(self, content_proj: np.ndarray, style_red: np.ndarray, sigma_diag_red: np.ndarray,
                 user_idx: np.ndarray, pre_sample: bool = True, n_versions: int = 4):
        """
        content_proj: [N, LATENT_DIM] PCA-projected content embeddings
        style_red: [n_users, LATENT_DIM] mu_128
        sigma_diag_red: [n_users, LATENT_DIM] per-user diagonal sigma in PCA space
        user_idx: [N] user index per content (sentence's user)
        pre_sample: if True, pre-sample n_versions s_u per content (cached)
        """
        self.content_proj = content_proj.astype(np.float32)
        self.style_red = style_red.astype(np.float32)
        self.sigma_diag_red = sigma_diag_red.astype(np.float32)
        self.user_idx = user_idx.astype(np.int64)
        self.pre_sample = pre_sample
        self.n_versions = n_versions
        if pre_sample:
            # Pre-sample s_u for all (content, version) pairs
            N = len(content_proj)
            self.s_u_cache = np.zeros((n_versions, N, LATENT_DIM), dtype=np.float32)
            for v in range(n_versions):
                for i in range(N):
                    u = user_idx[i]
                    mu = self.style_red[u]
                    sigma_diag = self.sigma_diag_red[u]
                    std = np.sqrt(np.maximum(sigma_diag, 1e-6))
                    self.s_u_cache[v, i] = mu + std * np.random.randn(LATENT_DIM).astype(np.float32)
            self._current_version = 0
            log(f"  pre-sampled s_u: cache shape {self.s_u_cache.shape}")

    def set_version(self, v: int):
        """Set which pre-sampled version to use (cycling across epochs)."""
        self._current_version = v % self.n_versions

    def __len__(self):
        return len(self.content_proj)

    def __getitem__(self, i: int):
        c = self.content_proj[i]
        if self.pre_sample:
            s_u = self.s_u_cache[self._current_version, i]
        else:
            u = self.user_idx[i]
            mu = self.style_red[u]
            sigma_diag = self.sigma_diag_red[u]
            std = np.sqrt(np.maximum(sigma_diag, 1e-6))
            s_u = mu + std * np.random.randn(LATENT_DIM).astype(np.float32)
        return c, s_u


def main():
    log("=" * 70)
    log("Phase 11.C: Conditional Latent Diffusion training")
    log("=" * 70)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    # === Load phase10 user_ids ===
    log("[1] Loading phase10 pairs (876 users) ...")
    phase10_users = []
    with PHASE10_PAIRS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            phase10_users.append(obj["user_id"])
    phase10_users = sorted(set(phase10_users))
    uid_to_idx = {u: i for i, u in enumerate(phase10_users)}
    log(f"  phase10 users: {len(phase10_users)}")

    # === Load phase11.A Gaussians ===
    log("[2] Loading Phase 11.A Gaussian cache ...")
    npz = np.load(GAUSSIANS_NPZ, allow_pickle=True)
    cached_uids = list(npz["user_ids"])
    cached_idx_for = {u: i for i, u in enumerate(cached_uids)}
    mu_768 = npz["mu_768"]                # [876, 768]
    sigma_diag = npz["sigma_diag"]        # [876, 768]
    pca_components = npz["pca_components"]  # [128, 768]
    pca_mean = npz["pca_mean"]            # [768]
    log(f"  cached users: {len(cached_uids)}")

    # === Project mu + sigma to PCA 128d ===
    log("[3] Projecting mu + sigma_diag to PCA 128d ...")
    mu_128 = (mu_768 - pca_mean) @ pca_components.T  # [876, 128]
    # sigma_diag in PCA: V_k^T diag(Sigma) V_k is non-diagonal,
    # but we use diagonal approximation: V_k^2 * sigma_diag (per-dim projection)
    # For sampling, this is a cheap but effective proxy.
    V_k = pca_components  # [128, 768]
    sigma_diag_128 = (V_k ** 2) @ sigma_diag.T  # [128, 876]
    sigma_diag_128 = sigma_diag_128.T.astype(np.float32)  # [876, 128]
    log(f"  mu_128: {mu_128.shape}, sigma_diag_128: {sigma_diag_128.shape}")

    # === Load phase10 train sentences ===
    log("[4] Loading phase10 train sentences ...")
    user_to_texts = {u: [] for u in phase10_users}
    user_to_sid = {u: [] for u in phase10_users}
    n_total = 0
    with RAW_SENTENCES_FILE.open() as f:
        sid_counter = 0
        for line in f:
            n_total += 1
            obj = json.loads(line)
            uid = obj["user_id"]
            if uid not in user_to_texts:
                continue
            if obj.get("is_holdout", False):
                continue
            txt = obj.get("sentence_text", "").strip()
            if not txt:
                continue
            user_to_texts[uid].append(txt)
            user_to_sid[uid].append(sid_counter)
            sid_counter += 1
    log(f"  total sentences: {n_total}, kept: {sid_counter}")

    # === Extract Qwen hidden states ===
    log("[5] Extracting Qwen hidden states (layer=14) for ~12k phase10 sentences ...")
    all_texts = []
    all_uids = []
    for uid in phase10_users:
        for t in user_to_texts[uid]:
            all_texts.append(t)
            all_uids.append(uid)
    log(f"  total texts to encode: {len(all_texts)}")

    # Cache file for content embeddings
    CACHE_CONTENT_EMBS = OUT_DIR / "phase11_c_content_embs_3584d.npy"
    CACHE_CONTENT_UIDS = OUT_DIR / "phase11_c_content_uids.npy"
    if CACHE_CONTENT_EMBS.exists() and CACHE_CONTENT_UIDS.exists():
        log(f"  Cache hit: {CACHE_CONTENT_EMBS}")
        content_embs_3584 = np.load(CACHE_CONTENT_EMBS)
        cached_uids_for_embs = list(np.load(CACHE_CONTENT_UIDS, allow_pickle=True))
        if len(cached_uids_for_embs) != len(all_uids) or any(a != b for a, b in zip(cached_uids_for_embs, all_uids)):
            log("  Cache mismatch, re-extracting ...")
            content_embs_3584 = None
        else:
            log(f"  loaded {content_embs_3584.shape}")
    else:
        content_embs_3584 = None

    if content_embs_3584 is None:
        # Import llm_client for Qwen2-7B
        sys.path.insert(0, str(REPO_ROOT))
        from llm_client import QwenLocalClient
        # Use QWEN_MODEL_PATH env var to point to actual weights path
        if "QWEN_MODEL_PATH" not in os.environ:
            os.environ["QWEN_MODEL_PATH"] = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
        client = QwenLocalClient(with_vllm=False)
        log(f"  Qwen client loaded (path={os.environ['QWEN_MODEL_PATH']})")

        t0 = time.time()
        result = client.get_hidden_states(
            all_texts,
            layers=[QWEN_LAYER_FOR_CONTENT],
            batch_size=QWEN_HIDDEN_BATCH,
            max_length=QWEN_HIDDEN_MAX_LEN,
        )
        content_embs_3584 = result[QWEN_LAYER_FOR_CONTENT].astype(np.float32)
        log(f"  extracted in {time.time()-t0:.1f}s, shape={content_embs_3584.shape}")

        np.save(CACHE_CONTENT_EMBS, content_embs_3584)
        np.save(CACHE_CONTENT_UIDS, np.array(all_uids, dtype=object))
        log(f"  cached")

    # === PCA 3584 -> 128 ===
    log("[6] PCA 3584 -> 128 on content embeddings ...")
    mu_c = content_embs_3584.mean(axis=0, keepdims=True)
    X_c = content_embs_3584 - mu_c
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    pca_c_components = Vt[:LATENT_DIM].astype(np.float32)  # [128, 3584]
    pca_c_mean = mu_c[0].astype(np.float32)  # [3584]
    explained_var_ratio = float((S[:LATENT_DIM] ** 2).sum() / (S ** 2).sum())
    log(f"  PCA explained variance: {explained_var_ratio:.4f} (top {LATENT_DIM})")

    # Project content to 128d
    content_embs_128 = (content_embs_3584 - pca_c_mean) @ pca_c_components.T  # [N, 128]
    content_embs_128 = content_embs_128.astype(np.float32)
    # Normalize to unit std (PCA-projected Qwen hiddens have std ~81, which destabilizes model)
    content_std = content_embs_128.std(axis=0, keepdims=True)  # [1, 128] per-dim std
    content_embs_128 = content_embs_128 / np.maximum(content_std, 1e-3)
    log(f"  content_embs_128 normalized: abs_max={np.abs(content_embs_128).max():.2f}")
    content_std_to_save = content_std.astype(np.float32)
    np.save(OUT_CONTENT_EMB, content_embs_128)
    # Save PCA components, mean, and per-dim std as a dict
    np.savez(OUT_CONTENT_PCA,
             pca_components=pca_c_components,
             pca_mean=pca_c_mean,
             content_std=content_std_to_save[0])
    log(f"  content_embs_128: {content_embs_128.shape}, content_std saved")

    # === Build dataset ===
    log("[7] Building dataset ...")
    user_idx_arr = np.array([uid_to_idx[u] for u in all_uids], dtype=np.int64)
    dataset = StyleContentDataset(
        content_proj=content_embs_128,
        style_red=mu_128.astype(np.float32),
        sigma_diag_red=sigma_diag_128,
        user_idx=user_idx_arr,
    )
    log(f"  dataset: {len(dataset)} samples")

    # === Train denoiser ===
    log(f"[8] Training denoiser ({N_EPOCHS} epochs, batch={BATCH_SIZE}, AMP={USE_AMP}) ...")
    device = DEVICE
    model = ConditionalDenoiser().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    ema = EMA(model, EMA_DECAY)
    diffusion = GaussianDiffusion(N_TIMESTEPS, BETA_SCHEDULE, device=device)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0, pin_memory=True, drop_last=True)
    n_steps = len(dataloader) * N_EPOCHS
    log(f"  total steps: {n_steps}, dataset={len(dataset)}")

    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    losses = []
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        if PRE_SAMPLE_S_U:
            dataset.set_version(epoch)
        model.train()
        epoch_loss = []
        for c, s_u in dataloader:
            c = c.to(device, non_blocking=True).float()
            s_u = s_u.to(device, non_blocking=True).float()
            B = c.size(0)
            z0 = c
            t = diffusion.sample_timesteps(B, device)
            noise = torch.randn_like(z0)
            z_t = diffusion.q_sample(z0, t, noise)
            with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=torch.bfloat16):
                eps_pred = model(z_t, t, c_raw=c, s_u=s_u)
                loss = F.mse_loss(eps_pred, noise)
            optimizer.zero_grad()
            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            ema.update(model)
            epoch_loss.append(loss.item())
        epoch_loss_mean = float(np.mean(epoch_loss))
        losses.append(epoch_loss_mean)
        if epoch % 3 == 0 or epoch == N_EPOCHS - 1:
            elapsed = time.time() - t0
            rate = (epoch + 1) / max(elapsed, 0.001)
            eta = (N_EPOCHS - epoch - 1) / max(rate, 0.001)
            log(f"  epoch {epoch:3d} loss={epoch_loss_mean:.4f} elapsed={elapsed:.1f}s ETA={eta:.0f}s")

    # === Save denoiser ===
    log("[9] Saving denoiser (EMA weights) ...")
    ema_model = ConditionalDenoiser().to(device)
    ema.apply_to(ema_model)
    ema_model.eval()
    torch.save({
        "state_dict": ema_model.state_dict(),
        "config": {
            "latent_dim": LATENT_DIM,
            "content_dim": CONTENT_DIM,
            "style_dim": LATENT_DIM,
            "hidden_dim": HIDDEN_DIFF_DIM,
            "n_blocks": N_DIFF_BLOCKS,
        },
    }, OUT_DENOISER)
    log(f"  saved: {OUT_DENOISER}")

    # === Validation: reconstruction + style transfer ===
    log("[10] Validation: reconstruction + style transfer ...")
    # Test 1: low-noise reconstruction (eps from clean, denoise)
    # Sample 100 sentences, get clean z_0 = c_proj, run reverse from t=10
    rng_v = np.random.default_rng(RANDOM_SEED + 1)
    sample_idxs = rng_v.choice(len(dataset), size=100, replace=False)
    cs = content_embs_128[sample_idxs]
    sus = np.stack([
        mu_128[uid_to_idx[all_uids[i]]] + np.sqrt(np.maximum(sigma_diag_128[uid_to_idx[all_uids[i]]], 1e-6)) * rng_v.standard_normal(LATENT_DIM).astype(np.float32)
        for i in sample_idxs
    ])
    cs_t = torch.from_numpy(cs).to(device)
    sus_t = torch.from_numpy(sus).to(device)

    # Single reverse step from t=10
    with torch.no_grad():
        t_low = torch.full((100,), 10, dtype=torch.long, device=device)
        noise_pred = ema_model(
            z_t=cs_t,
            t=t_low,
            c_raw=cs_t,
            s_u=sus_t,
        )
        z_recon = (cs_t - diffusion.sqrt_one_minus_alphas_cumprod[10] * noise_pred) / diffusion.sqrt_alphas_cumprod[10]
        recon_err = float(((z_recon - cs_t) ** 2).mean().sqrt().item())
        log(f"  Reconstruction error (t=10 reverse): {recon_err:.4f}")

    # Test 2: style transfer (sample A's content with B's style, see if closer to B's mu)
    # Use HIGH t where noise dominates — only s_u can guide recovery
    rng_st = np.random.default_rng(RANDOM_SEED + 2)
    pairs = []
    for _ in range(20):
        a_idx = int(rng_st.integers(0, len(dataset)))
        b_uid = int(rng_st.integers(0, len(phase10_users)))
        a_uid = user_idx_arr[a_idx]
        if a_uid == b_uid:
            b_uid = (b_uid + 1) % len(phase10_users)
        pairs.append((a_idx, b_uid))
    cos_to_b = []
    cos_to_a = []
    test_t_high = 500  # high noise, only s_u can guide
    with torch.no_grad():
        for a_idx, b_uid in pairs:
            c_a = torch.from_numpy(content_embs_128[a_idx:a_idx+1]).to(device)
            s_b = torch.from_numpy(mu_128[b_uid:b_uid+1]).to(device)
            s_a = torch.from_numpy(mu_128[user_idx_arr[a_idx]:user_idx_arr[a_idx]+1]).to(device)
            # Sample noisy z_t at high t
            noise = torch.randn_like(c_a)
            t_high = torch.full((1,), test_t_high, dtype=torch.long, device=device)
            z_t = diffusion.q_sample(c_a, t_high, noise)
            # Denoise with s_a vs s_b
            noise_pred_a = ema_model(z_t=z_t, t=t_high, c_raw=c_a, s_u=s_a)
            noise_pred_b = ema_model(z_t=z_t, t=t_high, c_raw=c_a, s_u=s_b)
            # Reconstruct x0
            sa_alpha = diffusion.sqrt_alphas_cumprod[test_t_high]
            so_alpha = diffusion.sqrt_one_minus_alphas_cumprod[test_t_high]
            z_recon_a = (z_t - so_alpha * noise_pred_a) / sa_alpha
            z_recon_b = (z_t - so_alpha * noise_pred_b) / sa_alpha
            mu_b = torch.from_numpy(mu_128[b_uid:b_uid+1]).to(device)
            mu_a = torch.from_numpy(mu_128[user_idx_arr[a_idx]:user_idx_arr[a_idx]+1]).to(device)
            cos_to_b.append(float(F.cosine_similarity(z_recon_b, mu_b, dim=-1).item()))
            cos_to_a.append(float(F.cosine_similarity(z_recon_b, mu_a, dim=-1).item()))
    style_transfer_pass = bool(np.mean(cos_to_b) > np.mean(cos_to_a))

    # Test 3: Does the model actually USE s_u? Compare outputs with s_a vs s_b
    diff_z = []
    with torch.no_grad():
        for a_idx, b_uid in pairs[:10]:
            c_a = torch.from_numpy(content_embs_128[a_idx:a_idx+1]).to(device)
            s_b = torch.from_numpy(mu_128[b_uid:b_uid+1]).to(device)
            s_a = torch.from_numpy(mu_128[user_idx_arr[a_idx]:user_idx_arr[a_idx]+1]).to(device)
            noise = torch.randn_like(c_a)
            t_high = torch.full((1,), test_t_high, dtype=torch.long, device=device)
            z_t = diffusion.q_sample(c_a, t_high, noise)
            noise_pred_a = ema_model(z_t=z_t, t=t_high, c_raw=c_a, s_u=s_a)
            noise_pred_b = ema_model(z_t=z_t, t=t_high, c_raw=c_a, s_u=s_b)
            diff_z.append(float((noise_pred_a - noise_pred_b).abs().mean().item()))
    s_u_signal = float(np.mean(diff_z))
    log(f"  s_u signal (avg abs diff in noise_pred between s_a vs s_b): {s_u_signal:.4f}")
    log(f"  Style transfer: cos(z_recon_b, mu_b)={np.mean(cos_to_b):.4f}, "
        f"cos(z_recon_b, mu_a)={np.mean(cos_to_a):.4f} "
        f"({'PASS' if style_transfer_pass else 'FAIL'})")

    # === Plot training curve ===
    log("[11] Saving training curve ...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Phase 11.C Training Loss")
        plt.tight_layout()
        plt.savefig(OUT_CURVE_PNG, dpi=100)
        log(f"  saved: {OUT_CURVE_PNG}")
    except Exception as e:
        log(f"  plot failed: {e}")

    # === Save meta ===
    meta = {
        "phase": "11.C",
        "latent_dim": LATENT_DIM,
        "content_dim": CONTENT_DIM,
        "n_users": len(phase10_users),
        "n_sentences": len(all_texts),
        "n_epochs": N_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "beta_schedule": BETA_SCHEDULE,
        "n_timesteps": N_TIMESTEPS,
        "pca_explained_variance_ratio": explained_var_ratio,
        "final_loss": float(losses[-1]),
        "recon_err_t10": recon_err,
        "style_transfer_pass": style_transfer_pass,
        "cos_z_recon_b_mu_b": float(np.mean(cos_to_b)),
        "cos_z_recon_b_mu_a": float(np.mean(cos_to_a)),
        "s_u_signal_abs_diff": s_u_signal,
        "ema_decay": EMA_DECAY,
        "qwen_layer": QWEN_LAYER_FOR_CONTENT,
        "n_diff_blocks": N_DIFF_BLOCKS,
        "hidden_dim": HIDDEN_DIFF_DIM,
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    log(f"  meta: {OUT_META}")

    log("=" * 70)
    log("PHASE 11.C COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()