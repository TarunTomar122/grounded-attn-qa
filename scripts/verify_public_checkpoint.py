from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from grounded_qa.needleish import NeedleConfig, NeedleishModel, load_public_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the public JAX and PyTorch Needle checkpoints.")
    parser.add_argument("--jax-repo", type=Path, required=True)
    parser.add_argument("--jax-checkpoint", type=Path, required=True)
    parser.add_argument("--safetensors", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.jax_repo))
    import jax
    import jax.numpy as jnp
    from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig, make_causal_mask, make_padding_mask

    with args.jax_checkpoint.open("rb") as handle:
        released = pickle.load(handle)
    jax_params = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.bfloat16), released["params"])
    jax_model = SimpleAttentionNetwork(TransformerConfig(**released["config"]))

    source = np.array([[71, 260, 991, 5, 402, 17, 2801, 6, 93]], dtype=np.int32)
    target = np.array([[1, 4, 103, 220, 0, 0, 0, 0]], dtype=np.int32)
    source_mask = make_padding_mask(jnp.asarray(source), 0)
    target_mask = make_causal_mask(target.shape[1])
    jax_memory, jax_cross_mask = jax_model.apply(
        {"params": jax_params}, jnp.asarray(source), src_mask=source_mask, method="encode"
    )
    jax_logits = jax_model.apply(
        {"params": jax_params},
        jnp.asarray(target),
        jax_memory,
        self_mask=target_mask,
        cross_mask=jax_cross_mask,
        method="decode",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_model = NeedleishModel(NeedleConfig.public_checkpoint()).to(dtype=torch.bfloat16)
    load_public_checkpoint(torch_model, args.safetensors)
    torch_model = torch_model.to(device).eval()
    source_t = torch.from_numpy(source).to(device)
    target_t = torch.from_numpy(target).to(device)
    with torch.inference_mode():
        torch_memory = torch_model.encode(source_t, source_t.ne(0))
        torch_logits = torch_model.decode(
            target_t,
            torch_memory,
            source_t.ne(0),
            torch.ones_like(target_t, dtype=torch.bool),
        )

    jax_memory_np = np.asarray(jax_memory, dtype=np.float32)
    jax_logits_np = np.asarray(jax_logits, dtype=np.float32)
    torch_memory_np = torch_memory.float().cpu().numpy()
    torch_logits_np = torch_logits.float().cpu().numpy()
    memory_delta = np.abs(jax_memory_np - torch_memory_np)
    logits_delta = np.abs(jax_logits_np - torch_logits_np)
    report = {
        "device": str(device),
        "encoder_max_abs": float(memory_delta.max()),
        "encoder_mean_abs": float(memory_delta.mean()),
        "logits_max_abs": float(logits_delta.max()),
        "logits_mean_abs": float(logits_delta.mean()),
        "jax_top_ids": np.argsort(jax_logits_np[0, 0])[-10:][::-1].tolist(),
        "torch_top_ids": np.argsort(torch_logits_np[0, 0])[-10:][::-1].tolist(),
    }
    print(report)
    if report["jax_top_ids"][:3] != report["torch_top_ids"][:3]:
        raise SystemExit("Checkpoint implementations disagree on the first three tokens")


if __name__ == "__main__":
    main()
