import random
import math
import numpy as np
import torch
from torch.utils.data import Sampler


class ClassAwareBatchSampler(Sampler):
    """
    Class-aware batch sampler ensuring balanced or controlled class representation
    per batch. Specifically designed for imbalanced datasets like UIT-VSFC where
    the Neutral class is severely underrepresented (~4.01%).

    Args:
        labels (list or np.ndarray): Ground truth labels for all training samples.
        batch_size (int): Total batch size (default: 32).
        class_proportions (dict or list): Target number of samples per class in each batch.
            For 3 classes (0: neg, 1: neu, 2: pos) with batch_size 32,
            a default configuration is {0: 13, 1: 6, 2: 13}.
        shuffle (bool): Whether to shuffle indices within each class per epoch.
        seed (int): Random seed for reproducibility.
    """

    def __init__(self, labels, batch_size=32, class_proportions=None, shuffle=True, seed=42):
        super().__init__()
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Unique classes and their sample indices
        self.classes = np.unique(self.labels)
        self.class_indices = {c: np.where(self.labels == c)[0].tolist() for c in self.classes}

        # Determine class distribution per batch
        if class_proportions is None:
            # Default for 3 classes: allocate at least 6 to minority class (1), rest split
            if batch_size >= 32:
                neu_count = 6
            elif batch_size >= 16:
                neu_count = 4
            else:
                neu_count = max(2, batch_size // 4)
            rem = batch_size - neu_count
            neg_count = rem // 2
            pos_count = rem - neg_count
            self.class_proportions = {0: neg_count, 1: neu_count, 2: pos_count}
        else:
            self.class_proportions = class_proportions

        assert sum(self.class_proportions.values()) == self.batch_size, (
            f"Sum of class proportions ({sum(self.class_proportions.values())}) "
            f"must equal batch_size ({self.batch_size})"
        )

        # Number of batches determined by majority class coverage
        max_class_len = max(len(self.class_indices[c]) for c in self.classes if c != 1)
        self.n_batches = math.ceil(max_class_len / max(self.class_proportions[0], self.class_proportions[2]))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        # Create pools of indices for each class
        pools = {}
        for c in self.classes:
            inds = list(self.class_indices[c])
            if self.shuffle:
                rng.shuffle(inds)
            pools[c] = inds

        cursors = {c: 0 for c in self.classes}

        for _ in range(self.n_batches):
            batch = []
            for c in self.classes:
                needed = self.class_proportions[c]
                cls_pool = pools[c]
                curr_cursor = cursors[c]

                if curr_cursor + needed <= len(cls_pool):
                    selected = cls_pool[curr_cursor : curr_cursor + needed]
                    cursors[c] += needed
                else:
                    # Wrap around
                    selected = cls_pool[curr_cursor:]
                    shortage = needed - len(selected)
                    if self.shuffle:
                        rng.shuffle(cls_pool)
                    selected.extend(cls_pool[:shortage])
                    cursors[c] = shortage

                batch.extend(selected)

            if self.shuffle:
                rng.shuffle(batch)

            yield batch

    def __len__(self):
        return self.n_batches
