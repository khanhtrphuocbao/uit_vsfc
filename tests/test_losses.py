import pytest
import torch
from src.losses import NeutralBoundaryLoss, SupervisedContrastiveLoss, compute_class_weights


def test_neutral_boundary_loss_normal():
    loss_fn = NeutralBoundaryLoss(margin=0.2, neutral_label=1, positive_mode="hardest")
    # Batch size 6, dim 16
    # Labels: 2 neutrals (1), 2 negatives (0), 2 positives (2)
    embeddings = torch.randn(6, 16, requires_grad=True)
    labels = torch.tensor([1, 0, 1, 2, 0, 2], dtype=torch.long)

    loss = loss_fn(embeddings, labels)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0.0

    # Test backward pass
    loss.backward()
    assert embeddings.grad is not None
    assert not torch.isnan(embeddings.grad).any()


def test_neutral_boundary_loss_no_neutral():
    loss_fn = NeutralBoundaryLoss(margin=0.2, neutral_label=1)
    embeddings = torch.randn(4, 16, requires_grad=True)
    labels = torch.tensor([0, 2, 0, 2], dtype=torch.long)  # No class 1

    loss = loss_fn(embeddings, labels)
    assert loss.item() == 0.0
    assert not torch.isnan(loss)

    loss.backward()
    assert embeddings.grad is not None
    assert torch.all(embeddings.grad == 0.0)


def test_neutral_boundary_loss_single_neutral():
    loss_fn = NeutralBoundaryLoss(margin=0.2, neutral_label=1)
    embeddings = torch.randn(4, 16, requires_grad=True)
    labels = torch.tensor([1, 0, 2, 0], dtype=torch.long)  # Only 1 neutral

    loss = loss_fn(embeddings, labels)
    assert loss.item() == 0.0
    assert not torch.isnan(loss)

    loss.backward()
    assert embeddings.grad is not None


def test_neutral_boundary_loss_all_neutral():
    loss_fn = NeutralBoundaryLoss(margin=0.2, neutral_label=1)
    embeddings = torch.randn(4, 16, requires_grad=True)
    labels = torch.tensor([1, 1, 1, 1], dtype=torch.long)  # No non-neutrals

    loss = loss_fn(embeddings, labels)
    assert loss.item() == 0.0
    assert not torch.isnan(loss)

    loss.backward()
    assert embeddings.grad is not None


def test_supcon_loss_normal():
    loss_fn = SupervisedContrastiveLoss(temperature=0.1)
    embeddings = torch.randn(6, 16, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1, 2, 2], dtype=torch.long)

    loss = loss_fn(embeddings, labels)
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0

    loss.backward()
    assert embeddings.grad is not None


def test_class_weights_computation():
    labels = [0] * 50 + [1] * 10 + [2] * 40
    weights = compute_class_weights(labels, num_classes=3, mode="inverse")
    assert len(weights) == 3
    # Minority class (1) must have highest weight
    assert weights[1] > weights[0]
    assert weights[1] > weights[2]
