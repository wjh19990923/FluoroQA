"""Artificial random images for plumbing tests only; no registration claims."""
import torch


class SyntheticProvider:
    def __init__(self, config):
        self.config = config

    def identity(self):
        return "synthetic-plumbing-v1"

    def _batches(self, seed):
        generator = torch.Generator().manual_seed(seed)
        n, r = self.config["batch_size"], self.config["resolution"]
        x = torch.rand(n, 1, r, r, generator=generator)
        yield dict(fixed=x, positive=x.clone(), negative=torch.rand(n, 1, r, r, generator=generator), mtre_mm=torch.linspace(.5, 5, n))

    def train_batches(self, epoch):
        return self._batches(self.config["seed"]+epoch)

    def val_batches(self):
        return self._batches(100000)


def build(config):
    return SyntheticProvider(config)
