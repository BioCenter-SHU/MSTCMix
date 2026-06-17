# utils/ASC.py
import random
import numpy as np
from torch.utils.data import Sampler


def _as_int_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return [int(v) for v in x]
    return [int(x)]


def compute_clean_count_from_gmm_dataset(GMM_dataset, global_indices, num_class: int):
    clean_count = np.zeros(int(num_class), dtype=np.int64)
    labels_oh = GMM_dataset.labels  # numpy [N, C]
    for gi in global_indices:
        c = int(np.argmax(labels_oh[int(gi)]))
        clean_count[c] += 1
    return clean_count


class ASCBatchSampler(Sampler):

    def __init__(
        self,
        global_indices,         
        GMM_dataset,
        num_class: int,
        batch_size: int,
        class_groups: dict,
        majority_classes=None,
        clean_count=None,
        anchor_gamma: float = 1.0,
        drop_last: bool = True,
        seed: int = 0,
    ):
        super().__init__(None)
        self.local_to_global = [int(i) for i in list(global_indices)]  # local_idx -> global_idx
        self.n_local = len(self.local_to_global)

        self.GMM_dataset = GMM_dataset
        self.C = int(num_class)
        self.batch_size = int(batch_size)
        self.class_groups = class_groups if class_groups is not None else {}
        self.majority = set(_as_int_list(majority_classes))
        self.anchor_gamma = float(anchor_gamma)
        self.drop_last = bool(drop_last)
        self.rng = random.Random(int(seed))

        if clean_count is None:
            clean_count = compute_clean_count_from_gmm_dataset(GMM_dataset, self.local_to_global, self.C)
        self.clean_count = np.asarray(clean_count, dtype=np.float64)

        labels_oh = self.GMM_dataset.labels  # numpy [N,C]
        self.local_to_class = {}
        for li, gi in enumerate(self.local_to_global):
            self.local_to_class[int(li)] = int(np.argmax(labels_oh[int(gi)]))

    def __len__(self):
        return self.n_local // self.batch_size


    def _build_pools(self):
        pools = {c: [] for c in range(self.C)}
        for li in range(self.n_local):
            c = self.local_to_class[int(li)]
            pools[c].append(int(li))
        for c in range(self.C):
            self.rng.shuffle(pools[c])
        return pools

    def _active_classes(self, pools):
        return [c for c in range(self.C) if len(pools[c]) > 0]

    def _only_majority_left(self, pools):
        active = self._active_classes(pools)
        if len(active) == 0:
            return True
        return all((c in self.majority) for c in active)

    def _weighted_anchor_class(self, pools):
        active = self._active_classes(pools)
        if len(active) == 0:
            return None

        weights = []
        for c in active:
            cc = float(self.clean_count[c]) if c < len(self.clean_count) else 0.0
            w = 1.0 / ((cc + 1.0) ** self.anchor_gamma)
            weights.append(w)

        s = float(sum(weights))
        if s <= 0:
            return self.rng.choice(active)

        r = self.rng.random() * s
        acc = 0.0
        for c, w in zip(active, weights):
            acc += w
            if acc >= r:
                return c
        return active[-1]

    def _pick_from_class(self, pools, c):
        if c is None or len(pools[c]) == 0:
            return None
        return pools[c].pop()  

    def _choose_min_epoch_count(self, candidates, epoch_count):
        if len(candidates) == 0:
            return None
        m = min(epoch_count[c] for c in candidates)
        mins = [c for c in candidates if epoch_count[c] == m]
        return self.rng.choice(mins)

    def _median_epoch_count(self, active, epoch_count):
        vals = [epoch_count[c] for c in active]
        vals.sort()
        n = len(vals)
        if n == 0:
            return 0
        if n % 2 == 1:
            return vals[n // 2]
        return (vals[n // 2 - 1] + vals[n // 2]) // 2

    def _sample_triplet(self, pools, epoch_count):
        if self._only_majority_left(pools):
            return None

        # (1) anchor
        a = self._weighted_anchor_class(pools)
        if a is None:
            return None
        idx_a = self._pick_from_class(pools, a)  # local_idx
        if idx_a is None:
            return None

        trip = [idx_a]
        epoch_count[a] += 1

        # (2) neighbor
        ga = self.class_groups.get(int(a), [])
        neigh_active = [int(c) for c in ga if int(c) != int(a) and len(pools[int(c)]) > 0]
        if len(neigh_active) > 0:
            nb = self._choose_min_epoch_count(neigh_active, epoch_count)
            idx_b = self._pick_from_class(pools, nb)
            if idx_b is not None:
                trip.append(idx_b)
                epoch_count[nb] += 1

        if self._only_majority_left(pools):
            return trip

        # (3) third
        active = self._active_classes(pools)
        if len(active) == 0:
            return trip

        mmed = self._median_epoch_count(active, epoch_count)
        target = int(mmed) + 1

        maj_active = [c for c in active if c in self.majority]
        maj_min = min(epoch_count[c] for c in maj_active) if len(maj_active) > 0 else None

        if maj_min is not None and maj_min < target:
            cand = [c for c in maj_active if epoch_count[c] == maj_min]
            c3 = self.rng.choice(cand)
        else:
            gmin = min(epoch_count[c] for c in active)
            cand = [c for c in active if epoch_count[c] == gmin]
            c3 = self.rng.choice(cand)

        idx_c = self._pick_from_class(pools, c3)
        if idx_c is not None:
            trip.append(idx_c)
            epoch_count[c3] += 1

        return trip

    def __iter__(self):
        pools = self._build_pools()
        epoch_count = [0 for _ in range(self.C)]

        while True:
            if self._only_majority_left(pools):
                break

            batch = []
            while len(batch) < self.batch_size:
                tri = self._sample_triplet(pools, epoch_count)
                if tri is None:
                    break
                batch.extend(tri)

                if len(batch) > self.batch_size:
                    batch = batch[: self.batch_size]
                    break

                if self._only_majority_left(pools):
                    break

            if len(batch) < self.batch_size:
                if self.drop_last:
                    break
                if len(batch) == 0:
                    break
                yield batch
                break

            yield batch