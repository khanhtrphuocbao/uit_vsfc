import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, class_names, title, save_path):
    """
    Generate and save a publication-quality confusion matrix image.
    """
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Class labels
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, pad=15)

    # Text annotations
    threshold = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            color = "white" if val > threshold else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot: {save_path}")


def plot_learning_curves(history, title, save_path):
    """
    Plot training loss and validation Macro-F1 across epochs.
    """
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_macro = [h["val_macro_f1"] for h in history]
    val_neu = [h["val_neutral_f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    # Loss curve
    ax1.plot(epochs, train_loss, "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, val_loss, "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Training & Validation Loss", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    # F1 curves
    ax2.plot(epochs, val_macro, "o-", label="Val Macro-F1", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, val_neu, "^-.", label="Val Neutral-F1", color="#d62728", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("F1 Score", fontsize=11)
    ax2.set_title("Validation F1 Progression", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved learning curves plot: {save_path}")


def main():
    class_names = ["Negative", "Neutral", "Positive"]
    figures_dir = "outputs/figures"
    os.makedirs(figures_dir, exist_ok=True)

    # Search for all result files
    result_files = glob.glob("outputs/**/result_seed_42.json", recursive=True)
    for rf in result_files:
        with open(rf, "r", encoding="utf-8") as f:
            data = json.load(f)

        exp_name = data["config"]["name"].replace("/", "_")
        cm = data["test_metrics"]["confusion_matrix"]
        plot_confusion_matrix(
            cm=cm,
            class_names=class_names,
            title=f"Confusion Matrix: {exp_name} (Seed 42)",
            save_path=os.path.join(figures_dir, f"cm_{exp_name}.png")
        )

        if "history" in data and len(data["history"]) > 1:
            plot_learning_curves(
                history=data["history"],
                title=f"Learning Curve: {exp_name} (Seed 42)",
                save_path=os.path.join(figures_dir, f"curve_{exp_name}.png")
            )


if __name__ == "__main__":
    main()
