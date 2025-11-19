import os
import argparse
from runner import Diffusion   # file chứa class Diffusion

def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Training / Sampling / FID")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "fid"],
                        help="train: train model, fid: generate images + compute FID")

    parser.add_argument("--log_path", type=str, default="./logs",
                        help="Folder lưu checkpoint và log")

    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Checkpoint để load khi mode=fid")

    parser.add_argument("--fake_folder", type=str, default="./images/fake",
                        help="Folder ảnh fake được sinh ra khi mode=fid")

    parser.add_argument("--real_folder", type=str, default="./images/real",
                        help="Folder chứa ảnh real dùng tính FID")

    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Số lượng ảnh fake khi tính FID")

    return parser.parse_args()


def main():
    args = parse_args()

    from config import config

    diffusion = Diffusion(args, config)


    if args.mode == "train":
        print("===== TRAINING MODE =====")
        diffusion.train()
        print("Training completed!")
        return

    # ============================================================
    # =============== 2. FID MODE ================================
    # ============================================================
    elif args.mode == "fid":
        print("===== SAMPLE + FID MODE =====")

        # Load model
        model = diffusion.load_model(ckpt_path=args.ckpt_path)

        # Make sure fake folder exists
        os.makedirs(args.fake_folder, exist_ok=True)

        # Step 1: Generate images
        print("Generating fake images...")
        diffusion.sample_fid(model, n_samples=args.n_samples)
        print(f"Generated {args.n_samples} images --> {args.fake_folder}")

        # Step 2: Compute FID
        print("Computing FID...")
        fid = compute_fid_from_folders(
            fake_folder=args.fake_folder,
            real_folder=args.real_folder
        )
        print(f"===== FID = {fid:.4f} =====")


if __name__ == "__main__":
    main()