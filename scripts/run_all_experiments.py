import os
import sys
import json
import argparse
import torch
from transformers import AutoTokenizer

from src.dataset import UITVSFCDataset
from src.trainer import run_training_experiment


def main():
    parser = argparse.ArgumentParser(description="Run complete experimental suite on UIT-VSFC")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2024, 777, 999],
                        help="List of random seeds (default: 5 seeds)")
    parser.add_argument("--epochs", type=int, default=4, help="Epochs per run")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--lambda_contrastive", type=float, default=0.1, help="Lambda for contrastive loss")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin for NBAC loss")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for SupCon")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    cache_snapshot = "/home/lavender/.cache/huggingface/hub/models--vinai--phobert-base/snapshots/01daacda68afe13d83023d16ec647239e344a1e6"
    model_path = cache_snapshot if os.path.exists(cache_snapshot) else "vinai/phobert-base"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_dataset = UITVSFCDataset("data/train.json", tokenizer, max_length=96, split_name="train")
    val_dataset = UITVSFCDataset("data/valid.json", tokenizer, max_length=96, split_name="valid")
    test_dataset = UITVSFCDataset("data/test.json", tokenizer, max_length=96, split_name="test")

    # Define all configurations to run
    # Format: (exp_id, loss_type, use_sampler, use_val_calibration)
    configurations = [
        ("B1_ce", "ce", False, False),
        ("B2_weighted_ce", "weighted_ce", False, False),
        ("B3_ce_val_bias", "ce", False, True),
        ("B4_ce_sampler", "ce", True, False),
        ("B5_supcon", "supcon", True, False),
        ("B6_nbac", "nbac", True, False),
        ("B7_nbac_val_bias", "nbac", True, True),
    ]

    total_runs = len(configurations) * len(args.seeds)
    print(f"\nTotal planned experimental runs: {total_runs} ({len(configurations)} configs x {len(args.seeds)} seeds)")

    run_idx = 0
    for exp_id, loss_type, use_sampler, use_calib in configurations:
        for seed in args.seeds:
            run_idx += 1
            print(f"\n=======================================================")
            print(f"[{run_idx}/{total_runs}] RUNNING {exp_id} | Seed {seed}")
            print(f"=======================================================")

            # Check if already completed
            res_file = os.path.join(args.output_dir, exp_id, f"result_seed_{seed}.json")
            if os.path.exists(res_file):
                print(f"Found existing result at {res_file}, skipping to save time.")
                continue

            config = {
                "name": exp_id,
                "seed": seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": 0.01,
                "max_length": 96,
                "model_path": model_path,
                "loss_type": loss_type,
                "use_sampler": use_sampler,
                "lambda_contrastive": args.lambda_contrastive,
                "margin": args.margin,
                "temperature": args.temperature,
                "use_val_calibration": use_calib,
            }

            run_training_experiment(
                config=config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                device=device,
                output_dir=args.output_dir,
            )

    print("\nALL EXPERIMENTS COMPLETED!")


if __name__ == "__main__":
    main()
