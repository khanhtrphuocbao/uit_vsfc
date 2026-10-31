import os
import sys
import argparse
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.getcwd())

import torch
from transformers import AutoTokenizer

from src.dataset import UITVSFCDataset
from src.trainer import run_training_experiment


def main():
    parser = argparse.ArgumentParser(description="Train PhoBERT models on UIT-VSFC")
    parser.add_argument("--exp", type=str, required=True,
                        choices=["B1_ce", "B2_weighted_ce", "B3_ce_val_bias",
                                 "B4_ce_sampler", "B5_supcon", "B6_nbac", "B7_nbac_val_bias",
                                 "pilot_nbac"],
                        help="Experiment identifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_length", type=int, default=96, help="Max sequence length")
    parser.add_argument("--lambda_contrastive", type=float, default=0.1, help="Contrastive loss weight")
    parser.add_argument("--margin", type=float, default=0.2, help="NBAC margin")
    parser.add_argument("--temperature", type=float, default=0.1, help="SupCon temperature")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.0f} MB)")

    # Model path (using local snapshot directory if available)
    cache_snapshot = "/home/lavender/.cache/huggingface/hub/models--vinai--phobert-base/snapshots/01daacda68afe13d83023d16ec647239e344a1e6"
    model_path = cache_snapshot if os.path.exists(cache_snapshot) else "vinai/phobert-base"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading datasets...")
    train_dataset = UITVSFCDataset("data/train.json", tokenizer, max_length=args.max_length, split_name="train")
    val_dataset = UITVSFCDataset("data/valid.json", tokenizer, max_length=args.max_length, split_name="valid")
    test_dataset = UITVSFCDataset("data/test.json", tokenizer, max_length=args.max_length, split_name="test")
    print(f"Loaded train ({len(train_dataset)}), valid ({len(val_dataset)}), test ({len(test_dataset)})")

    # Experiment mapping
    config = {
        "name": args.exp,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "model_path": model_path,
        "lambda_contrastive": args.lambda_contrastive,
        "margin": args.margin,
        "temperature": args.temperature,
    }

    if args.exp == "B1_ce":
        config.update({
            "loss_type": "ce",
            "use_sampler": False,
            "use_val_calibration": False,
        })
    elif args.exp == "B2_weighted_ce":
        config.update({
            "loss_type": "weighted_ce",
            "use_sampler": False,
            "use_val_calibration": False,
        })
    elif args.exp == "B3_ce_val_bias":
        config.update({
            "loss_type": "ce",
            "use_sampler": False,
            "use_val_calibration": True,
        })
    elif args.exp == "B4_ce_sampler":
        config.update({
            "loss_type": "ce",
            "use_sampler": True,
            "use_val_calibration": False,
        })
    elif args.exp == "B5_supcon":
        config.update({
            "loss_type": "supcon",
            "use_sampler": True,
            "use_val_calibration": False,
        })
    elif args.exp == "B6_nbac":
        config.update({
            "loss_type": "nbac",
            "use_sampler": True,
            "use_val_calibration": False,
        })
    elif args.exp == "B7_nbac_val_bias":
        config.update({
            "loss_type": "nbac",
            "use_sampler": True,
            "use_val_calibration": True,
        })
    elif args.exp == "pilot_nbac":
        config.update({
            "name": f"pilot_nbac_lam{args.lambda_contrastive}_m{args.margin}",
            "loss_type": "nbac",
            "use_sampler": True,
            "use_val_calibration": False,
        })

    result = run_training_experiment(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        device=device,
        output_dir=args.output_dir,
    )

    print("\nExperiment completed successfully.")
    return result


if __name__ == "__main__":
    main()
