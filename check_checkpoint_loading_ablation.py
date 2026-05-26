import tempfile
from pathlib import Path

import torch

from RainFormerPhys_ablation import VALID_VARIANTS, get_rainformer_model
from test_rainformer_ablation_LOOCV import load_checkpoint_with_variant_validation


def main():
    device = torch.device("cpu")
    seq_len = 400

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ckpts = {}

        # same-variant save/load must pass
        for variant in sorted(VALID_VARIANTS):
            model = get_rainformer_model(variant, in_channels=1, seq_len=seq_len)
            ckpt_path = tmpdir / f"{variant}.pth"
            torch.save({"variant": variant, "state_dict": model.state_dict()}, ckpt_path)
            ckpts[variant] = ckpt_path

            load_model = get_rainformer_model(variant, in_channels=1, seq_len=seq_len)
            load_checkpoint_with_variant_validation(load_model, str(ckpt_path), variant, device)
            print(f"[pass] same variant load ok: {variant}")

        # cross-variant load should fail with clear mismatch
        variants = sorted(VALID_VARIANTS)
        src_variant = variants[0]
        dst_variant = variants[1]
        dst_model = get_rainformer_model(dst_variant, in_channels=1, seq_len=seq_len)
        try:
            load_checkpoint_with_variant_validation(dst_model, str(ckpts[src_variant]), dst_variant, device)
        except RuntimeError as e:
            if "variant mismatch" in str(e).lower() or "checkpoint variant mismatch" in str(e).lower():
                print(f"[pass] cross variant mismatch correctly rejected: {src_variant} -> {dst_variant}")
                print(f"[info] message: {e}")
                return
            raise

        raise RuntimeError("Expected cross-variant loading to fail, but it succeeded.")


if __name__ == "__main__":
    main()
