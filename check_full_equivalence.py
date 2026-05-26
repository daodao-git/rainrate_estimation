import argparse

import torch

from RainFormerPhys import RainFormerPhys as OriginalRainFormerPhys
from RainFormerPhys_ablation import count_trainable_parameters, get_rainformer_model


def main(args):
    orig = OriginalRainFormerPhys(in_channels=args.in_channels, seq_len=args.seq_len, embed_dim=args.embed_dim, num_blocks=args.num_blocks)
    abl = get_rainformer_model("full", in_channels=args.in_channels, seq_len=args.seq_len, embed_dim=args.embed_dim, num_blocks=args.num_blocks)

    orig_items = list(orig.state_dict().items())
    abl_items = list(abl.state_dict().items())

    ok = True
    if len(orig_items) != len(abl_items):
        print(f"[mismatch] state_dict length: original={len(orig_items)} vs full={len(abl_items)}")
        ok = False

    max_len = min(len(orig_items), len(abl_items))
    for i in range(max_len):
        on, ov = orig_items[i]
        an, av = abl_items[i]
        if on != an:
            print(f"[mismatch] param name index {i}: original='{on}' vs full='{an}'")
            ok = False
        if tuple(ov.shape) != tuple(av.shape):
            print(f"[mismatch] shape for '{on}': original={tuple(ov.shape)} vs full={tuple(av.shape)}")
            ok = False

    orig_params = count_trainable_parameters(orig)
    abl_params = count_trainable_parameters(abl)
    print(f"[info] original param_count={orig_params}")
    print(f"[info] full-ablation param_count={abl_params}")
    if orig_params != abl_params:
        print("[mismatch] trainable parameter count differs")
        ok = False

    x = torch.randn(args.batch_size, args.in_channels, args.seq_len)
    y1 = orig(x)
    y2 = abl(x)
    print(f"[info] original output shape={tuple(y1.shape)}")
    print(f"[info] full-ablation output shape={tuple(y2.shape)}")
    if tuple(y1.shape) != tuple(y2.shape):
        print("[mismatch] output shape differs")
        ok = False

    if ok:
        print("[pass] full variant is structurally equivalent to original RainFormerPhys.")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check structural equivalence between original RainFormerPhys and ablation full variant.")
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=400)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    main(parser.parse_args())
