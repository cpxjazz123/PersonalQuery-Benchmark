#!/usr/bin/env python3
"""Stage 04b — Trainable per-user diagonal Gaussian via reparameterization.

Input (same source as fit_per_user_gaussian.py):
  - pcfg_cache/strict3_embeddings.npz (Stage 03 output 80/20 z split)
  - pcfg_cache/user_n_sents.json

Output:
  - result/04_gaussian/user_gaussian_stats_trainable.json
      schema:
        config: {schema_version=3, covariance="trainable_diag",
                 syntax_dim, latent_dim=16, kl_beta, n_epochs, lr,
                 init_from="empirical_diag", note}
        users[uid] = {
            mu:  list[float len=16]    # trainable center
            log_sigma2: list[float len=16]  # trainable log-precision
            sigma_inv_diag: list[float len=16]  # 1/sigma^2 (aligned with full Sigma schema)
            sigma_diag: list[float len=16]
            n, n_val,
            d2_q50, d2_q75, d2_q95, d2_max: empirical val d^2 quantile
            d2_q95_theoretical, d2_q05_theoretical,
            d2_q50_theoretical, d2_q75_theoretical: chi2(16, q)
        }

Training objective:
    L = GaussianNLL(z_profile | mu_u, sigma_u) + beta * KL(N(mu_u, sigma_u^2) || N(0, I))

Design:
    - sigma_u uses diagonal (16-dim log sigma^2_d); not learning full 16x16 covariance:
      * 21k users x 256-element full cov = 5.4M params, unstable and overfits
      * Diagonal form is consistent with downstream d2 = sum ((z-mu)/sigma)^2
    - Initialization: mu_u <- empirical mean, log_sigma2_u <- log(empirical var + eps)
    - Only profile sentences enter training; val sentences only used to evaluate d^2 quantiles
    - d2 qXX still from val empirical statistics (preserves fall-back consistent with fit_per_user_gaussian)
    - d2_qXX_theoretical still given by chi2(16, q) (consistent with current schema)

Usage:
    nohup $PQ_PYTHON \
        04_gaussian/trainable_per_user_gaussian.py \
        > $PQ_SCRATCH/trainable_gaussian.log 2>&1 &
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import os
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("PQ_SCRATCH", str(REPO_ROOT / "scratch")))
CACHE_DIR = SCRATCH / "pcfg_cache"
# User instruction 2026-09-23: one per category (Baby / Musical / Video_Games),
# main() runs 3 domains serially, writes outputs to result/04_gaussian/<subdir>/.
CATEGORY_INPUTS = [
    # (category_key, subdir)
    ("Baby",                "baby"),
    ("Musical_Instruments", "musical"),
    ("Video_Games",         "video_games"),
]
OUT_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats_trainable.json"
# Allow env var to override output path (for ablation experiments)
if "TG_OUT_PATH" in __import__("os").environ:
    OUT_PATH = Path(__import__("os").environ["TG_OUT_PATH"])

# ---- Hardcoded configuration (Rule 3) ----
SCHEMA_VERSION = 3
# Embedding source: 'strict3' (default, 16d) or 'svd_mlp' (64d, SVD+MLP pipeline)
# Switched via TG_SOURCE env var; switching source lets the same trainable Gaussian
# pipeline consume either the strict3 supervised encoder or the svd_mlp two-step
# dimensionality-reduction encoder, avoiding accumulating multiple main-script
# versions under 04_gaussian/ (Rule 3)
import os as _os
SOURCE = _os.environ.get('TG_SOURCE', 'strict3')
NPZ_BY_SOURCE = {
    'strict3': 'strict3_embeddings.npz',
    'svd_mlp': 'svd_mlp_embeddings.npz',
}
if SOURCE not in NPZ_BY_SOURCE:
    raise ValueError(f"unknown TG_SOURCE={SOURCE!r}; valid: {list(NPZ_BY_SOURCE)}")
NPZ_NAME = NPZ_BY_SOURCE[SOURCE]
# Latent dim: svd_mlp defaults to 64d, strict3 fixed at 16d; main() later re-validates against npz
LATENT_DIM = 64 if SOURCE == 'svd_mlp' else 16

_KL_BETA_ENV = _os.environ.get('TG_KL_BETA')
KL_BETA = float(_KL_BETA_ENV) if _KL_BETA_ENV is not None else 1e-3  # KL coefficient; overridable via TG_KL_BETA
N_EPOCHS = 20              # 20 epochs usually sufficient for convergence
BATCH_USERS = 4096         # number of samples per training step
SENTS_PER_USER_CAP = 200   # per-user profile subsample cap, controls batch tensor size
LR = 3e-3
WEIGHT_DECAY = 0.0
SIGMA2_FLOOR = 1e-4        # numerical floor, prevents sigma^2 -> 0 from blowing up NLL
_LS2_OFF_ENV = _os.environ.get('TG_LOG_SIGMA2_OFFSET')
LOG_SIGMA2_INIT_OFFSET = float(_LS2_OFF_ENV) if _LS2_OFF_ENV is not None else 0.0
_SEED_ENV = _os.environ.get('TG_SEED')
SEED = int(_SEED_ENV) if _SEED_ENV is not None else 42
MIN_PROFILE_SENTS = 40     # consistent with fit_per_user_gaussian
MIN_VAL_SENTS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# chi2(LATENT_DIM, q) is computed dynamically at main() startup based on current LATENT_DIM
CHI2_THEORETICAL: dict = {}


def _compute_chi2_table(df: int) -> dict:
    """Returns chi2(df, q) theoretical quantile table for d2_qXX_theoretical fields."""
    from scipy.stats import chi2
    return {
        "q05": float(chi2.ppf(0.05, df)),
        "q50": float(chi2.ppf(0.50, df)),
        "q75": float(chi2.ppf(0.75, df)),
        "q95": float(chi2.ppf(0.95, df)),
    }


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_split() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Loads profile/val split for the current SOURCE (default strict3, switchable to svd_mlp).

    Returns: (uid_list, z_profile_sorted, z_val_sorted, prof_offsets, val_offsets, z_dim)
    z_dim comes from the npz 'z_dim' field (svd_mlp 64, strict3 16); used later to validate LATENT_DIM.
    """
    npz_path = CACHE_DIR / NPZ_NAME
    n_sents_path = CACHE_DIR / "user_n_sents.json"
    with open(n_sents_path) as f:
        user_n_sents = [int(n) for n in json.load(f)]
    npz = np.load(npz_path, allow_pickle=True)  # svd_mlp contains object uid_list
    uid_list = [str(u) for u in npz["uid_list"]]
    z_profile = np.asarray(npz["z_profile"], dtype=np.float32)
    z_val = np.asarray(npz["z_val"], dtype=np.float32)
    profile_idx = np.asarray(npz["profile_idx"], dtype=np.int64)
    val_idx = np.asarray(npz["val_idx"], dtype=np.int64)
    # Validate latent dim: svd_mlp stores z_dim explicitly; strict3 falls back to 16
    if "z_dim" in npz.files:
        z_dim = int(np.asarray(npz["z_dim"]).item())
    else:
        z_dim = 16
    if z_dim != LATENT_DIM:
        raise ValueError(
            f"SOURCE={SOURCE} expected LATENT_DIM={LATENT_DIM}, "
            f"but {NPZ_NAME} has z_dim={z_dim}"
        )

    uid_idx_per_row = np.repeat(
        np.arange(len(uid_list), dtype=np.int64), user_n_sents
    )
    prof_uid = uid_idx_per_row[profile_idx]
    val_uid = uid_idx_per_row[val_idx]
    prof_order = np.argsort(prof_uid, kind="stable")
    val_order = np.argsort(val_uid, kind="stable")
    prof_counts = np.bincount(prof_uid, minlength=len(uid_list))
    val_counts = np.bincount(val_uid, minlength=len(uid_list))
    prof_offsets = np.concatenate(
        ([0], np.cumsum(prof_counts, dtype=np.int64))
    )
    val_offsets = np.concatenate(
        ([0], np.cumsum(val_counts, dtype=np.int64))
    )
    log(
        f"loaded: N={len(uid_list)} users, "
        f"z_profile={z_profile.shape}, z_val={z_val.shape}"
    )
    return uid_list, z_profile[prof_order], z_val[val_order], prof_offsets, val_offsets, z_dim


def compute_init_params(
    uid_list: list[str],
    z_profile: np.ndarray,
    z_val: np.ndarray,
    prof_offsets: np.ndarray,
    val_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Initializes mu and log_sigma2 with empirical moments."""
    N = len(uid_list)
    D = LATENT_DIM
    mu_init = np.zeros((N, D), dtype=np.float32)
    log_sigma2_init = np.zeros((N, D), dtype=np.float32)
    eligible = []
    for u in range(N):
        ps, pe = prof_offsets[u], prof_offsets[u + 1]
        vs, ve = val_offsets[u], val_offsets[u + 1]
        n_p, n_v = pe - ps, ve - vs
        if n_p < MIN_PROFILE_SENTS or n_v < MIN_VAL_SENTS:
            continue
        zp = z_profile[ps:pe]
        mu = zp.mean(axis=0)
        var = zp.var(axis=0)
        var = np.maximum(var, SIGMA2_FLOOR)
        mu_init[u] = mu
        if _os.environ.get('TG_SIGMA_INIT', 'empirical') == 'one':
            log_sigma2_init[u] = 0.0  # log(1) = 0 -> sigma=1 in each dim
        elif _os.environ.get('TG_SIGMA_INIT') == 'big':
            log_sigma2_init[u] = 1.0  # log(e) ~ 1 -> sigma~2.7 in each dim
        else:
            log_sigma2_init[u] = np.log(var) + LOG_SIGMA2_INIT_OFFSET
        eligible.append(u)
    log(
        f"init params: eligible={len(eligible)}/{N} users "
        f"(min_profile={MIN_PROFILE_SENTS}, min_val={MIN_VAL_SENTS})"
    )
    return mu_init, log_sigma2_init, eligible


class PerUserDiagGaussian(nn.Module):
    """One diagonal Gaussian per user N(mu_u, diag(sigma_u^2)); trainable parameters."""

    def __init__(self, n_users: int, latent_dim: int,
                 mu_init: np.ndarray, log_sigma2_init: np.ndarray):
        super().__init__()
        self.mu = nn.Embedding(n_users, latent_dim)
        self.log_sigma2 = nn.Embedding(n_users, latent_dim)
        with torch.no_grad():
            self.mu.weight.copy_(torch.from_numpy(mu_init))
            self.log_sigma2.weight.copy_(torch.from_numpy(log_sigma2_init))

    def get_params(self, u_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mu(u_idx), self.log_sigma2(u_idx)

    def nll(self, z: torch.Tensor, u_idx: torch.Tensor) -> torch.Tensor:
        """Per-sample Gaussian NLL (diagonal).

        NLL = 0.5 * sum_d [ (z-mu)^2 / sigma^2 + log sigma^2 ]
        """
        mu, log_sigma2 = self.get_params(u_idx)
        sigma2 = torch.exp(log_sigma2).clamp_min(SIGMA2_FLOOR)
        diff2 = (z - mu) ** 2 / sigma2
        log_sigma2_safe = torch.log(sigma2)
        nll = 0.5 * (diff2.sum(dim=-1) + log_sigma2_safe.sum(dim=-1))
        return nll

    def kl_to_prior(self, u_idx: torch.Tensor) -> torch.Tensor:
        """KL( N(mu_u, diag(sigma_u^2)) || N(0, I) ) per user."""
        mu, log_sigma2 = self.get_params(u_idx)
        sigma2 = torch.exp(log_sigma2).clamp_min(SIGMA2_FLOOR)
        kl = 0.5 * (mu ** 2 + sigma2 - 1.0 - log_sigma2).sum(dim=-1)
        return kl


class UserSentenceDataset(Dataset):
    """Returns (z, uid) samples for training.

    profile sentences are flattened as rows; numpy views are used to save memory.
    """

    def __init__(self, z_profile: np.ndarray, prof_offsets: np.ndarray,
                 eligible: list[int], sents_cap: int, rng: np.random.Generator):
        rows: list[tuple[int, int]] = []
        for u in eligible:
            ps, pe = prof_offsets[u], prof_offsets[u + 1]
            n = pe - ps
            if n > sents_cap:
                idx = rng.choice(n, size=sents_cap, replace=False)
                rows.extend((ps + int(i), u) for i in idx)
            else:
                rows.extend((ps + i, u) for i in range(n))
        self.row_uid = np.asarray(rows, dtype=np.int64)
        self.z = z_profile

    def __len__(self) -> int:
        return len(self.row_uid)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        row, uid = self.row_uid[i]
        return self.z[row], int(uid)


def evaluate_val_d2(
    model: PerUserDiagGaussian,
    uid_list: list[str],
    eligible: list[int],
    z_val: np.ndarray,
    val_offsets: np.ndarray,
) -> dict[str, dict]:
    """Computes d^2 quantiles on val sentences using trainable parameters."""
    model.eval()
    out: dict[str, dict] = {}
    n_done = 0
    with torch.no_grad():
        for u in eligible:
            vs, ve = val_offsets[u], val_offsets[u + 1]
            zv = torch.from_numpy(z_val[vs:ve]).to(DEVICE)
            mu, log_sigma2 = model.get_params(torch.tensor([u], device=DEVICE))
            mu = mu.squeeze(0)
            log_sigma2 = log_sigma2.squeeze(0)
            sigma2 = torch.exp(log_sigma2).clamp_min(SIGMA2_FLOOR)
            d2 = ((zv - mu) ** 2 / sigma2).sum(dim=-1).cpu().numpy()
            d2_sorted = np.sort(d2)
            n_v = d2_sorted.size
            q50 = float(d2_sorted[int(n_v * 0.50)])
            q75 = float(d2_sorted[int(n_v * 0.75)])
            q95 = float(d2_sorted[min(int(n_v * 0.95), n_v - 1)])
            dmax = float(d2_sorted[-1])
            mu_np = mu.cpu().numpy()
            sigma2_np = sigma2.cpu().numpy()
            n_profile = int(
                # prof_offsets is not passed in explicitly; n is set to 0 here for
                # schema consistency, and the real profile n is rewritten by the caller
                0
            )
            out[uid_list[u]] = {
                "mu": mu_np.astype(np.float32).tolist(),
                "log_sigma2": log_sigma2.cpu().numpy().astype(np.float32).tolist(),
                "sigma_diag": sigma2_np.astype(np.float32).tolist(),
                "sigma_inv_diag": (1.0 / sigma2_np).astype(np.float32).tolist(),
                "n": n_profile,
                "n_val": int(n_v),
                "d2_q50": q50,
                "d2_q75": q75,
                "d2_q95": q95,
                "d2_max": dmax,
                "d2_q95_theoretical": CHI2_THEORETICAL["q95"],
                "d2_q05_theoretical": CHI2_THEORETICAL["q05"],
                "d2_q50_theoretical": CHI2_THEORETICAL["q50"],
                "d2_q75_theoretical": CHI2_THEORETICAL["q75"],
            }
            n_done += 1
            if n_done % 2000 == 0:
                log(f"  evaluated {n_done}/{len(eligible)}")
    return out


def main_task_body() -> None:
    global CHI2_THEORETICAL
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    log(f"device={DEVICE}  source={SOURCE}  npz={NPZ_NAME}  latent_dim={LATENT_DIM}")

    uid_list, z_profile, z_val, prof_offsets, val_offsets, z_dim = load_split()
    # Populate module-level dict so evaluate_val_d2 can read it via globals
    CHI2_THEORETICAL.clear()
    CHI2_THEORETICAL.update(_compute_chi2_table(z_dim))
    N = len(uid_list)
    D = LATENT_DIM
    assert z_profile.shape[1] == D, (
        f"{NPZ_NAME} dim {z_profile.shape[1]} != LATENT_DIM {D}"
    )

    mu_init, log_sigma2_init, eligible = compute_init_params(
        uid_list, z_profile, z_val, prof_offsets, val_offsets,
    )
    if not eligible:
        raise RuntimeError("no eligible users")

    model = PerUserDiagGaussian(N, D, mu_init, log_sigma2_init).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    rng = np.random.default_rng(SEED)

    n_profile_sents = sum(
        prof_offsets[u + 1] - prof_offsets[u] for u in eligible
    )
    log(f"profile sents (eligible) = {n_profile_sents}")

    t0 = time.time()
    for epoch in range(N_EPOCHS):
        ds = UserSentenceDataset(
            z_profile, prof_offsets, eligible,
            sents_cap=SENTS_PER_USER_CAP, rng=rng,
        )
        loader = DataLoader(
            ds, batch_size=BATCH_USERS, shuffle=True, num_workers=0,
            drop_last=True, pin_memory=(DEVICE == "cuda"),
        )
        epoch_nll = 0.0
        epoch_kl = 0.0
        n_batches = 0
        model.train()
        for z, u_idx in loader:
            z = z.to(DEVICE, non_blocking=True)
            u_idx = u_idx.to(DEVICE, non_blocking=True)
            nll = model.nll(z, u_idx).mean()
            kl = model.kl_to_prior(u_idx).mean()
            loss = nll + KL_BETA * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_nll += float(nll.detach())
            epoch_kl += float(kl.detach())
            n_batches += 1
        avg_nll = epoch_nll / max(n_batches, 1)
        avg_kl = epoch_kl / max(n_batches, 1)
        log(
            f"epoch {epoch + 1:02d}/{N_EPOCHS}  "
            f"nll={avg_nll:.4f}  kl={avg_kl:.4f}  "
            f"loss={avg_nll + KL_BETA * avg_kl:.4f}  "
            f"elapsed={time.time() - t0:.1f}s"
        )

    log("evaluating val d^2 quantiles with trained params...")
    users_out = evaluate_val_d2(
        model, uid_list, eligible, z_val, val_offsets,
    )

    # Fill in the n field (profile sentence count)
    for u in eligible:
        users_out[uid_list[u]]["n"] = int(
            prof_offsets[u + 1] - prof_offsets[u]
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {
            "schema_version": SCHEMA_VERSION,
            "covariance": "trainable_diag",
            "syntax_dim": D,
            "latent_dim": D,
            "source": SOURCE,
            "source_npz": NPZ_NAME,
            "kl_beta": KL_BETA,
            "n_epochs": N_EPOCHS,
            "batch_users": BATCH_USERS,
            "sents_per_user_cap": SENTS_PER_USER_CAP,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "sigma2_floor": SIGMA2_FLOOR,
            "seed": SEED,
            "min_profile_sents": MIN_PROFILE_SENTS,
            "min_val_sents": MIN_VAL_SENTS,
            "init_from": "empirical_diag",
            "n_users_total": N,
            "n_users_eligible": len(eligible),
            "device": DEVICE,
            "note": (
                f"trainable per-user diagonal Gaussian via reparameterization; "
                f"mu/log_sigma2 are trainable Embedding(N,{LATENT_DIM}); trained with "
                f"GaussianNLL + KL_beta*KL(N(mu,sigma^2)||N(0,I)) on profile sentences; "
                f"d2 quantiles are empirical val-set Mahalanobis under TRAINED sigma; "
                f"theoretical chi2({LATENT_DIM}, q) values retained for downstream fallback; "
                f"embedding source = {SOURCE} ({NPZ_NAME})."
            ),
        },
        "users": users_out,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f)
    log(
        f"DONE wrote {OUT_PATH}  users={len(users_out)}/{N}  "
        f"total_time={time.time() - t0:.1f}s"
    )


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    """User instruction 2026-09-23: run 3 categories serially.

    For each category, rebind the script's path constants to category-specific paths,
    then call the original main_task_body() (keep original logic untouched).
    Outputs are written to result/<stage>/<baby|musical|video_games>/ subdirectories.
    """
    global SENT_CACHE, UID_TO_SENTS, ASIN_USERS_PATH, ATTRIBUTES_PATH, META_FILE, OUT_DIR, OUT_PATH  # noqa
    # backup current (Baby) defaults
    saved = {
        k: v for k, v in globals().items()
        if k in {"SENT_CACHE", "UID_TO_SENTS", "ASIN_USERS_PATH", "ATTRIBUTES_PATH",
                 "META_FILE", "OUT_DIR", "OUT_PATH"}
        and isinstance(v, Path)
    }
    base_out = REPO_ROOT / "result" / Path(__file__).parent.name
    for category, subdir in CATEGORY_INPUTS:
        log(f"\n========== [{category}] (subdir={subdir}) ==========")
        # Reset all known category-dependent paths to point at the per-category subdir.
        if "SENT_CACHE" in saved:
            SENT_CACHE = REPO_ROOT / "result/02_user_review_sentence_extract" / f"uid_to_sentences_{subdir}.pkl"
        if "UID_TO_SENTS" in saved:
            UID_TO_SENTS = REPO_ROOT / "result/02_user_review_sentence_extract" / f"uid_to_sentences_{subdir}.pkl"
        if "ASIN_USERS_PATH" in saved:
            ASIN_USERS_PATH = REPO_ROOT / "result/02_user_review_sentence_extract" / f"asin_to_users_{subdir}.pkl"
        if "ATTRIBUTES_PATH" in saved:
            ATTRIBUTES_PATH = REPO_ROOT / "result/01_attribute_extraction" / f"product_attributes_{subdir}.pkl"
        if "META_FILE" in saved:
            META_FILE = Path(os.environ.get("PQ_DATA_DIR", str(REPO_ROOT / "data"))) / {
                "baby": "meta_Baby_Products_2023.jsonl",
                "musical": "meta_Musical_Instruments.jsonl",
                "video_games": "meta_Video_Games.jsonl",
            }[subdir]
        if "OUT_DIR" in saved:
            OUT_DIR = base_out / subdir
        if "OUT_PATH" in saved:
            OUT_PATH = base_out / subdir / saved["OUT_PATH"].name
        OUT_DIR.mkdir(parents=True, exist_ok=True) if "OUT_DIR" in saved else None
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True) if "OUT_PATH" in saved else None
        try:
            main_task_body()
        except Exception as e:
            log(f"[{category}] FAILED: {e!r}")
            raise
    # Restore Baby defaults (for import compatibility with downstream).
    for k, v in saved.items():
        globals()[k] = v


if __name__ == "__main__":
    main()
