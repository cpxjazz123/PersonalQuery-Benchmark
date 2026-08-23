#!/usr/bin/env python3
"""Build stub user_style_vectors_real.jsonl for 10K users (with STRICT_ALPHA=0).

With ALPHA=0.0, the actual injected vector values are multiplied by zero, so
the file just needs to exist with the right shape (3584-dim). We pad the
VADES 20-dim latent mu with zeros so the file is structurally correct.

Reads: /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/vades_10k_user_profiles.jsonl
       /home/wlia0047/ar57/wenyu/PersoanlQuery/result/query_records_10k.json (for uid list)
Writes: /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/user_style_vectors_real.jsonl
"""
from __future__ import annotations
import json
from pathlib import Path

SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
REPO = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")

USER_PROFILES = SCRATCH / "vades_10k_user_profiles.jsonl"
RECORDS = REPO / "result/query_records_10k.json"
OUT = Path("/home/wlia0047/hj82_scratch2/wenyu/user_style_steering") / "user_style_vectors_real.jsonl"

LAYERS = [16, 20, 24, 26]


def main() -> None:
    # Load VADES profiles
    profiles: dict[str, list[float]] = {}
    with open(USER_PROFILES, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            profiles[r["user_id"]] = r["user_mu"]
    print(f"[stub] loaded {len(profiles)} VADES profiles")

    # Load target user_ids from records
    records = json.load(open(RECORDS, "r", encoding="utf-8"))
    target_uids = {r["user_id"] for r in records}
    print(f"[stub] target users from records: {len(target_uids)}")

    # Use the first profile to know the hidden_dim target
    sample_uid = next(iter(profiles))
    latent_dim = len(profiles[sample_uid])
    print(f"[stub] latent_dim from VADES profiles: {latent_dim}")

    # We need 3584-dim vector per layer. Since STRICT_ALPHA=0.0, values don't matter.
    # We'll write a 3584-dim zero vector (could also write the latent_dim-dim vector
    # padded with zeros — but 3584-dim is the safe size that matches Qwen hidden).
    HIDDEN_DIM = 3584

    out_rows: list[dict] = []
    for uid in target_uids:
        v_latent = profiles.get(uid)
        row = {"user_id": uid}
        for L in LAYERS:
            # Provide a 3584-dim vector; ALPHA=0 makes the actual values irrelevant.
            # We'll keep it as zeros (cleanest).
            row[f"residual_mean_layer_{L}"] = [0.0] * HIDDEN_DIM
        out_rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[stub] wrote {len(out_rows)} rows -> {OUT}")
    print(f"[stub] (zero vectors; ALPHA=0 makes values irrelevant)")


if __name__ == "__main__":
    main()
