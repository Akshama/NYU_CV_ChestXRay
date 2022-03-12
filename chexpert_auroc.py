import sys
import torch

import sklearn.metrics as metrics

from chexpert_data import ChexpertDataset, chexpert_collate, DATA_LABELS
from chexpert_main import set_device
from pathlib import Path
from tqdm import tqdm

def eval():
    BATCH_SIZE = 1
    NUM_CLASSES = 14
    device = set_device(True)

    base = Path("/scratch/ahm9968/chexpert-small/")
    mdlpath = Path("/scratch/ahm9968/multirun/2021-11-03/23-14-05/4/model.best_valid")

    valid = torch.utils.data.DataLoader(
        ChexpertDataset(base, train=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
        collate_fn=chexpert_collate,
    )

    model = torch.load(mdlpath, map_location=device)

    y_truths = []
    y_scores = []
    with tqdm(total=len(valid), desc="Validating: ", file=sys.stdout) as pbar:
        for i, (xs, ys) in enumerate(valid):
            preds = model(xs.to(device))
            y_truths.append(ys[3])
            y_scores.append(preds[3].detach())
            pbar.update(1)
    
    y_truths = torch.cat(y_truths).view(-1, NUM_CLASSES)
    y_scores = torch.cat(y_scores).view(-1, NUM_CLASSES).sigmoid().to("cpu")
    
    for i in range(y_truths.size(1)):
        idx = i + len(DATA_LABELS) - NUM_CLASSES
        try:
            print("{}: {:.4f}".format(DATA_LABELS[idx], metrics.roc_auc_score(y_truths[:, i], y_scores[:, i])))
        except ValueError:
            print("No positive labels for {}, so no AUC".format(DATA_LABELS[idx]))

if __name__ == "__main__":
    eval()