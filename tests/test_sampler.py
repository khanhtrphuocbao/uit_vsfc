import numpy as np
from src.sampler import ClassAwareBatchSampler


def test_sampler_batch_distribution():
    # Simulated dataset with extreme imbalance:
    # Class 0: 500, Class 1: 40 (Neutral), Class 2: 500
    labels = [0] * 500 + [1] * 40 + [2] * 500
    batch_size = 32
    target_proportions = {0: 13, 1: 6, 2: 13}

    sampler = ClassAwareBatchSampler(
        labels=labels,
        batch_size=batch_size,
        class_proportions=target_proportions,
        shuffle=True,
        seed=42,
    )

    batches = list(sampler)
    assert len(batches) > 0

    for batch in batches:
        assert len(batch) == batch_size
        batch_labels = [labels[i] for i in batch]
        counts = {c: batch_labels.count(c) for c in [0, 1, 2]}
        assert counts[0] == 13
        assert counts[1] == 6
        assert counts[2] == 13


def test_sampler_reproducibility():
    labels = [0] * 100 + [1] * 10 + [2] * 100
    s1 = ClassAwareBatchSampler(labels, batch_size=16, class_proportions={0: 6, 1: 4, 2: 6}, seed=123)
    s2 = ClassAwareBatchSampler(labels, batch_size=16, class_proportions={0: 6, 1: 4, 2: 6}, seed=123)

    b1 = list(s1)
    b2 = list(s2)
    assert b1 == b2
