import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from src.losses import NeutralBoundaryLoss, SupervisedContrastiveLoss, BalancedSoftmaxLoss


class PhoBertSentimentClassifier(nn.Module):
    """
    PhoBERT-based sentiment classifier supporting standard Cross-Entropy,
    Class-Weighted Cross-Entropy, Supervised Contrastive Loss, and
    Neutral-Boundary-Aware Contrastive Loss (NBAC).

    Args:
        model_name_or_path (str): Hugging Face model identifier or local snapshot directory.
        num_classes (int): Number of target sentiment classes (default: 3).
        dropout_rate (float): Dropout probability for classification head (default: 0.1).
        loss_type (str): 'ce', 'weighted_ce', 'supcon', 'nbac'.
        class_weights (Tensor or None): Optional class weights tensor for weighted_ce.
        lambda_contrastive (float): Weight of contrastive loss term (default: 0.1).
        margin (float): Separation margin for NBAC loss (default: 0.2).
        temperature (float): Temperature for SupCon loss (default: 0.1).
    """

    def __init__(
        self,
        model_name_or_path="vinai/phobert-base",
        num_classes=3,
        dropout_rate=0.1,
        loss_type="ce",
        class_weights=None,
        lambda_contrastive=0.1,
        margin=0.2,
        temperature=0.1,
        neutral_label=1,
        class_counts=None,
        gamma=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.lambda_contrastive = lambda_contrastive

        # Resolve local snapshot or fallback
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        try:
            self.roberta = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        except Exception:
            self.roberta = AutoModel.from_pretrained(model_name_or_path, config=self.config, use_safetensors=True)

        hidden_size = self.config.hidden_size

        # Classification Head (matches standard RoBERTa classification head)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Loss functions
        if loss_type in ["balanced_softmax", "balanced_supcon"]:
            counts = class_counts if class_counts is not None else [5325, 458, 5643]
            self.ce_loss_fn = BalancedSoftmaxLoss(counts, gamma=gamma)
        elif loss_type in ["weighted_ce", "weighted_supcon"] and class_weights is not None:
            self.ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss_fn = nn.CrossEntropyLoss()

        if loss_type == "nbac":
            self.boundary_loss_fn = NeutralBoundaryLoss(
                margin=margin,
                neutral_label=neutral_label,
                positive_mode="hardest"
            )
        elif loss_type in ["supcon", "balanced_supcon", "weighted_supcon"]:
            self.contrastive_loss_fn = SupervisedContrastiveLoss(temperature=temperature)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        # Use [CLS] representation (token 0)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        projected = self.activation(self.dense(cls_rep))
        features = projected  # Semantic representation h_i before classifier
        dropped = self.dropout(features)
        logits = self.classifier(dropped)

        total_loss = None
        ce_loss_val = None
        contrast_loss_val = None

        if labels is not None:
            ce_loss_val = self.ce_loss_fn(logits, labels)

            if self.loss_type == "nbac":
                contrast_loss_val = self.boundary_loss_fn(features, labels)
                total_loss = ce_loss_val + self.lambda_contrastive * contrast_loss_val
            elif self.loss_type in ["supcon", "balanced_supcon", "weighted_supcon"]:
                contrast_loss_val = self.contrastive_loss_fn(features, labels)
                total_loss = ce_loss_val + self.lambda_contrastive * contrast_loss_val
            else:
                total_loss = ce_loss_val
                contrast_loss_val = torch.tensor(0.0, device=logits.device)

        return {
            "loss": total_loss,
            "ce_loss": ce_loss_val,
            "contrast_loss": contrast_loss_val,
            "logits": logits,
            "features": features,
        }
