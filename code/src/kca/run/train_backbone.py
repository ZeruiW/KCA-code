"""Train the Simple4DNet reference backbone on a 4D action task (CPU-friendly).

This produces a genuinely *learned* model to explain (vs. the hand-coded oracle),
so KCA can be validated on a real network. On the GPU host the same loop trains the
real backbones via their own scripts; here it gives an end-to-end, runnable example.
"""
from __future__ import annotations

import torch

from ..types import Clip
from ..data.base import Dataset
from ..backbones.simple4d import Simple4DNet, Simple4DBackbone


def collate(dataset: Dataset, idxs):
    pts = torch.cat([dataset.get(i).clip.points for i in idxs], 0)
    mask = torch.cat([dataset.get(i).clip.mask for i in idxs], 0)
    label = torch.cat([dataset.get(i).clip.label for i in idxs], 0)
    return Clip(points=pts, mask=mask, label=label)


@torch.no_grad()
def accuracy(net, dataset, idxs, bs=32):
    net.eval(); correct = total = 0
    for s in range(0, len(idxs), bs):
        b = collate(dataset, idxs[s:s + bs])
        pred = net(b).argmax(1)
        correct += (pred == b.label).sum().item(); total += b.B
    return correct / max(total, 1)


def train(dataset: Dataset, epochs: int = 12, bs: int = 32, lr: float = 2e-3,
          dim: int = 64, heads: int = 4, layers: int = 2, grid: int = 3,
          val_dataset: Dataset | None = None, seed: int = 0, verbose: bool = True):
    torch.manual_seed(seed)
    net = Simple4DNet(dataset.num_classes, dim=dim, heads=heads, layers=layers, grid=grid)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(dataset)
    order = list(range(n))
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n).tolist()
        tot = 0.0
        for s in range(0, n, bs):
            idxs = perm[s:s + bs]
            b = collate(dataset, idxs)
            logits = net(b)
            loss = torch.nn.functional.cross_entropy(logits, b.label, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idxs)
        if verbose:
            tr = accuracy(net, dataset, order)
            msg = f"epoch {ep+1:2d} loss {tot/n:.3f} train-acc {tr:.3f}"
            if val_dataset is not None:
                msg += f" val-acc {accuracy(net, val_dataset, list(range(len(val_dataset)))):.3f}"
            print(msg)
    return Simple4DBackbone(net, dataset.num_classes)
