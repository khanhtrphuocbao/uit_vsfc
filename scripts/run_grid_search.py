import os
import json
import torch
from transformers import AutoTokenizer
from src.dataset import UITVSFCDataset
from src.trainer import run_training_experiment


def main():
    print("=" * 65)
    print("NBAC HYPERPARAMETER GRID SEARCH ON VALIDATION SET")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_snapshot = "/home/lavender/.cache/huggingface/hub/models--vinai--phobert-base/snapshots/01daacda68afe13d83023d16ec647239e344a1e6"
    model_path = cache_snapshot if os.path.exists(cache_snapshot) else "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    train_dataset = UITVSFCDataset("data/train.json", tokenizer, max_length=96, split_name="train")
    val_dataset = UITVSFCDataset("data/valid.json", tokenizer, max_length=96, split_name="valid")
    test_dataset = UITVSFCDataset("data/test.json", tokenizer, max_length=96, split_name="test")

    # Defined grid from prompt:
    # lambda in {0.05, 0.1, 0.2}, margin in {0.1, 0.2, 0.3}
    grid = [
        {"lambda": 0.05, "margin": 0.2},
        {"lambda": 0.1,  "margin": 0.1},
        {"lambda": 0.1,  "margin": 0.2},
        {"lambda": 0.1,  "margin": 0.3},
        {"lambda": 0.2,  "margin": 0.2},
    ]

    os.makedirs("outputs/grid_search", exist_ok=True)
    results = []

    best_val_macro = -1.0
    best_params = None

    for i, p in enumerate(grid, 1):
        lam = p["lambda"]
        m = p["margin"]
        tag = f"grid_nbac_lam{lam}_m{m}"
        print(f"\n[{i}/{len(grid)}] Evaluating: lambda={lam}, margin={m}")

        config = {
            "name": f"grid/{tag}",
            "seed": 42,
            "epochs": 4,
            "batch_size": 32,
            "lr": 2e-5,
            "weight_decay": 0.01,
            "max_length": 96,
            "model_path": model_path,
            "loss_type": "nbac",
            "use_sampler": True,
            "lambda_contrastive": lam,
            "margin": m,
            "use_val_calibration": False,
        }

        res = run_training_experiment(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            device=device,
            output_dir="outputs",
        )

        val_macro = res["best_val_metrics"]["macro_f1"]
        val_neu_f1 = res["best_val_metrics"]["neutral"]["f1"]
        results.append({
            "lambda": lam,
            "margin": m,
            "val_macro_f1": val_macro,
            "val_neutral_f1": val_neu_f1,
            "val_accuracy": res["best_val_metrics"]["accuracy"],
            "best_epoch": res["best_epoch"],
        })

        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_params = p

        print(f"--> [lambda={lam}, margin={m}] Val Macro-F1: {val_macro:.4f} | Val Neu-F1: {val_neu_f1:.4f}")

    print("\n" + "=" * 65)
    print(f"GRID SEARCH COMPLETE. Best on Validation: lambda={best_params['lambda']}, margin={best_params['margin']} (Val Macro-F1 = {best_val_macro:.4f})")
    print("=" * 65)

    with open("outputs/grid_search/grid_results.json", "w", encoding="utf-8") as f:
        json.dump({"best_params": best_params, "all_results": results}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
