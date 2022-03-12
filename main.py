
import logging
import os
import torch
import wandb

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def set_device(cuda : bool):
    device = "cpu"
    if cuda:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            log.info("no gpu found, defaulting to cpu")
    return torch.device(device)


def train(flags : DictConfig):
    pass


def valid(flags : DictConfig):
    pass


def main(flags : DictConfig):
    device = set_device(flags.cuda)
    torch.manual_seed(flags.random_seed)
    print("This is just a template.")

if __name__ == "__main__":
    main()