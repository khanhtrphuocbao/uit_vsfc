import torch
import torch.nn as nn
import torch.nn.functional as F


class NeutralBoundaryLoss(nn.Module):
    """
    Neutral-Boundary-Aware Contrastive Loss (NBAC Loss).

    Forces neutral representations to maintain a safety margin 'm' from the
    hardest polarizing samples (negative or positive) relative to positive neutral pairs:
        L_NB = (1 / |N|) * sum_{i in N} max(0, m - s(h_i, h_p_i) + s(h_i, h_q_i))

    where:
        - h_i is a normalized representation: h_i = z_i / ||z_i||_2
        - s(a, b) = a^T b is cosine similarity
        - p_i is a positive sample (another Neutral in the batch)
        - q_i is a hard negative (Negative or Positive sample in the batch with highest cosine similarity)
        - m is the margin parameter

    Args:
        margin (float): Separation margin (default: 0.2).
        neutral_label (int): Integer label representing the Neutral class (default: 1).
        positive_mode (str): 'hardest' (batch-hard min similarity) or 'mean' (mean positive similarity).
    """

    def __init__(self, margin=0.2, neutral_label=1, positive_mode="hardest"):
        super().__init__()
        self.margin = margin
        self.neutral_label = neutral_label
        self.positive_mode = positive_mode

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings (Tensor): Shape (B, D), unnormalized or normalized hidden representations.
            labels (Tensor): Shape (B,), ground truth class labels.

        Returns:
            Tensor: Scalar loss tensor with valid gradient history.
        """
        device = embeddings.device
        # Normalize representations to unit sphere: h_i = z_i / ||z_i||_2
        h = F.normalize(embeddings, p=2, dim=1)  # (B, D)

        # Pairwise cosine similarity matrix: s(a, b) = h_a^T h_b
        sim_matrix = torch.matmul(h, h.T)  # (B, B)

        # Identify Neutral and non-Neutral indices
        neutral_mask = (labels == self.neutral_label)
        non_neutral_mask = ~neutral_mask

        neutral_indices = torch.nonzero(neutral_mask, as_tuple=True)[0]
        non_neutral_indices = torch.nonzero(non_neutral_mask, as_tuple=True)[0]

        # Edge cases:
        # 1. No Neutral samples in batch
        # 2. Fewer than 2 Neutral samples (cannot form positive pair)
        # 3. No non-Neutral samples (cannot form hard negative)
        if len(neutral_indices) < 2 or len(non_neutral_indices) == 0:
            return torch.tensor(0.0, device=device, dtype=embeddings.dtype, requires_grad=True) + 0.0 * h.sum()

        losses = []
        for i in neutral_indices:
            # Positive candidates: other Neutral samples (excluding anchor i itself)
            pos_mask = neutral_mask.clone()
            pos_mask[i] = False
            pos_sims = sim_matrix[i, pos_mask]

            if len(pos_sims) == 0:
                continue

            if self.positive_mode == "hardest":
                # Hardest positive: lowest cosine similarity among neutrals
                s_pos = torch.min(pos_sims)
            else:
                s_pos = torch.mean(pos_sims)

            # Hard negative: highest cosine similarity among non-neutrals (Neg or Pos)
            neg_sims = sim_matrix[i, non_neutral_mask]
            s_neg = torch.max(neg_sims)

            # Triplet margin loss: max(0, m - s(h_i, h_p) + s(h_i, h_q))
            loss_i = F.relu(self.margin - s_pos + s_neg)
            losses.append(loss_i)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device, dtype=embeddings.dtype, requires_grad=True) + 0.0 * h.sum()

        loss = torch.stack(losses).mean()
        return loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Standard Supervised Contrastive Loss (SupCon) from Khosla et al. (NeurIPS 2020).
    Used as Baseline B5 to contrast against class-specific Neutral Boundary Loss.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        device = embeddings.device
        h = F.normalize(embeddings, p=2, dim=1)
        sim = torch.matmul(h, h.T) / self.temperature

        # For numerical stability
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        batch_size = embeddings.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Mask out self-contrast cases (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Mean log-likelihood over positive pairs
        mask_pos_pairs = mask.sum(1)
        valid_anchors = mask_pos_pairs > 0

        if not valid_anchors.any():
            return torch.tensor(0.0, device=device, dtype=embeddings.dtype, requires_grad=True) + 0.0 * h.sum()

        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_anchors] / mask_pos_pairs[valid_anchors]
        loss = -mean_log_prob_pos.mean()
        return loss


class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax Loss for long-tailed imbalanced classification (Ren et al., NeurIPS 2020; Menon et al., ICLR 2021).
    Shifts the logits during training by adding gamma * log(n_c) to eliminate the majority prior bias:
        L_BS = -log( exp(z_y + gamma * log(n_y)) / sum_c exp(z_c + gamma * log(n_c)) )

    Args:
        class_counts (list or Tensor): Number of training samples per class [n_0, n_1, ..., n_{C-1}].
        gamma (float): Prior adjustment exponent (default: 1.0; 0.5 for smoothed adjustment).
    """

    def __init__(self, class_counts, gamma=1.0):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        self.gamma = gamma
        log_prior = self.gamma * torch.log(counts + 1e-7)
        self.register_buffer("log_prior", log_prior)

    def forward(self, logits, labels):
        adjusted_logits = logits + self.log_prior.to(logits.device)
        return F.cross_entropy(adjusted_logits, labels)


def compute_class_weights(labels, num_classes=3, mode="inverse"):
    """
    Compute class weights for Cross-Entropy loss.
    mode='inverse': w_c = N / (C * N_c)
    """
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()
    total = len(labels)
    if mode == "inverse":
        weights = total / (num_classes * counts)
    elif mode == "sqrt_inverse":
        weights = torch.sqrt(total / (num_classes * counts))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return weights
