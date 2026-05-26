import torch

from RainFormerPhys_ablation import VALID_VARIANTS, count_trainable_parameters, get_rainformer_model


def main():
    x = torch.randn(4, 1, 400)
    for variant in sorted(VALID_VARIANTS):
        model = get_rainformer_model(variant, in_channels=1, seq_len=400, embed_dim=128, num_blocks=4)
        params = count_trainable_parameters(model)
        try:
            y = model(x)
            if tuple(y.shape) != (4,):
                raise RuntimeError(f"Output shape mismatch for {variant}: got {tuple(y.shape)}, expected (4,)")
            print(f"[pass] variant={variant:12s} params={params:9d} output_shape={tuple(y.shape)}")
        except Exception as e:
            print(f"[fail] variant={variant}: {e}")
            raise


if __name__ == "__main__":
    main()
