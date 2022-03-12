
import logging
import sys
import timeit
import torch
import wandb

import sklearn.metrics as metrics
import torch.nn.functional as F

from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm

from chexpert_data import ChexpertDataset, chexpert_collate, NUM_CLASSES, DATA_LABELS
from chexpert_model import Densenet121, ResNet152


log = logging.getLogger(__name__)


def set_device(cuda : bool):
    device = "cpu"
    if cuda:
        if torch.cuda.is_available():
            device = "cuda"
            log.info("found cuda, utilizing gpu")
        else:
            log.info("no gpu found, defaulting to cpu")
    return torch.device(device)


def binary_loss(preds, targets):
    return F.binary_cross_entropy_with_logits(preds, targets.float())


def pathology_loss(preds, targets, flag, targets_ones):
     if(flag == "ignore"):
         mask = targets >= 0
         preds = preds.masked_select(mask)
         targets = targets.masked_select(mask)
         return F.binary_cross_entropy_with_logits(preds, targets.float())
     else:
         targets = torch.where(targets == -1, targets_ones, targets)
         mask = targets >= 0
         preds = preds.masked_select(mask)
         targets = targets.masked_select(mask)
         return F.binary_cross_entropy_with_logits(preds, targets.float())

def regression_loss(preds, targets):
    return F.smooth_l1_loss(preds, targets)


def multiclass_loss(preds, targets):
    pass  # TODO


def train(flags : DictConfig, model, loader, device, epoch, optimizer):
    total_loss = 0
    sex_loss = 0
    age_loss = 0
    ang_loss = 0
    pth_loss = 0

    start = timeit.default_timer()
    
    with tqdm(total=len(loader), desc="Training Epoch {}: ".format(epoch), file=sys.stdout) as pbar:
        for i, (xs, ys) in enumerate(loader):
            if flags.debug >= 0 and i > flags.debug:
                break
            preds = model(xs.to(device))
            loss_sex = binary_loss(preds[0], ys[0].to(device))
            loss_age = regression_loss(preds[1], ys[1].to(device))
            loss_ang = binary_loss(preds[2], ys[2].to(device))
            targets_ones = torch.ones(ys[3].shape).to(device) 
            loss_pth = pathology_loss(preds[3], ys[3].to(device), flags.uncertainty, targets_ones)
            loss = 0
            if flags.chex_auxtask:
                loss += loss_sex + loss_age + loss_ang
            loss += loss_pth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            sex_loss += loss_sex.item()
            age_loss += loss_age.item()
            ang_loss += loss_ang.item()
            pth_loss += loss_pth.item()

            pbar.update(1)
        
    avg_sex_loss = float(sex_loss / len(loader))
    avg_age_loss = float(age_loss / len(loader))
    avg_ang_loss = float(ang_loss / len(loader))
    avg_pth_loss = float(pth_loss / len(loader))

    if flags.wandb:
        wandb.log({
            "epoch": epoch,
            "train_sex_loss": avg_sex_loss,
            "train_age_loss": avg_age_loss,
            "train_angle_loss": avg_ang_loss,
            "train_path_loss": avg_pth_loss,
        })
    log.info('Trained Epoch: {} | Avg Sex Loss: {:.6f}, Avg Age loss: {:.6f}, Avg Angle loss: {:.6f}, Avg Pathology loss: {:6f}'.format(
        epoch, avg_sex_loss, avg_age_loss, avg_ang_loss, avg_pth_loss))



def valid(flags : DictConfig, model, loader, device, epoch):
    total_loss = 0
    sex_loss = 0
    age_loss = 0
    ang_loss = 0

    y_truths = []  # pathologies
    y_scores = []
    with tqdm(total=len(loader), desc="Validating: ", file=sys.stdout) as pbar:
        for i, (xs, ys) in enumerate(loader):
            preds = model(xs.to(device))
            loss_sex = binary_loss(preds[0], ys[0].to(device))
            loss_age = regression_loss(preds[1], ys[1].to(device))
            loss_ang = binary_loss(preds[2], ys[2].to(device))
            sex_loss += loss_sex.item()
            age_loss += loss_age.item()
            ang_loss += loss_ang.item()

            preds = model(xs.to(device))
            y_truths.append(ys[3])
            y_scores.append(preds[3].detach())
            pbar.update(1)

    y_truths = torch.cat(y_truths).view(-1, NUM_CLASSES)  # on cpu
    y_scores = torch.cat(y_scores).view(-1, NUM_CLASSES).sigmoid().detach()
    targets_ones = torch.ones(y_truths.shape).to(device)
    pth_loss = pathology_loss(y_scores, y_truths.to(device), flags.uncertainty, targets_ones)
    y_truths, y_scores = y_truths.cpu(), y_scores.cpu()
    
    aucs = {}
    for i in range(y_truths.size(1)):
        idx = i + len(DATA_LABELS) - NUM_CLASSES
        try:
            aucs[DATA_LABELS[idx]] = metrics.roc_auc_score(y_truths[:, i], y_scores[:, i])
        except ValueError:
            # log.warning("No positive labels for {}, so no AUC".format(DATA_LABELS[idx]))
            pass
    average_auc = sum(aucs.values()) / len(aucs)


    avg_sex_loss = float(sex_loss / len(loader))
    avg_age_loss = float(age_loss / len(loader))
    avg_ang_loss = float(ang_loss / len(loader))
    avg_pth_loss = float(pth_loss / len(loader))

    if flags.wandb:
        wandb.log({
            "epoch": epoch,
            "valid_sex_loss": avg_sex_loss,
            "valid_age_loss": avg_age_loss,
            "valid_angle_loss": avg_ang_loss,
            "valid_path_loss": avg_pth_loss,
            "average_auc": average_auc,
            "auc_atelec": aucs["Atelectasis"],
            "auc_cardio": aucs["Cardiomegaly"],
            "auc_consol": aucs["Consolidation"],
            "auc_edema": aucs["Edema"],
            "auc_pleural": aucs["Pleural Effusion"],
        })
    log.info('Valid | Ave AUC: {:.4f}, Avg Pathology loss: {:.6f}, Avg Sex Loss: {:.6f}, Avg Age loss: {:.6f}, Avg Angle loss: {:.6f}, '.format(
        average_auc, avg_pth_loss, avg_sex_loss, avg_age_loss, avg_ang_loss))
    return total_loss


def main(flags : DictConfig):
    device = set_device(flags.cuda)
    torch.manual_seed(flags.random_seed)

    trainloader = torch.utils.data.DataLoader(
        ChexpertDataset(flags.chexpert_data, True),
        batch_size=flags.bs,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        collate_fn=chexpert_collate,
    )
    valloader = torch.utils.data.DataLoader(
        ChexpertDataset(flags.chexpert_data, False),
        batch_size=flags.bs,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        collate_fn=chexpert_collate,
    )
    

    best_valid_path = Path('model.best_valid')
    if best_valid_path.exists():
        model = torch.load(best_valid_path)
    elif flags.model == "densenet":
        model = Densenet121(flags.pretrained)
    elif flags.model == "resnet":
        model = ResNet152(flags.pretrained)
    else:
        raise RuntimeError("Did not recognize model {}".format(flags.model))
    model = model.to(device)

    if flags.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=flags.lr, momentum=flags.momentum, weight_decay=flags.weight_decay)
    elif flags.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=flags.lr, betas=(flags.beta1, flags.beta2), weight_decay=flags.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)


    best_valid = None
    retries = 0
    patience = 8
    ep = 0
    while retries < patience:
        ep += 1
        if ep > flags.max_epochs:
            log.info("Reached max_epochs, stopping training...")
            break
        train(flags, model, trainloader, device, ep, optimizer)
        valid_loss = valid(flags, model, valloader, device, ep)
        scheduler.step(valid_loss)
        retries += 1
        if best_valid is None or valid_loss < best_valid:
            retries = 0
            torch.save(model, best_valid_path)
            log.info("New best validation loss! Saved best valid model to {}".format(str(best_valid_path)))
            best_valid = valid_loss
    
    if retries == patience:
        log.info("Validation error has not improved in {} epochs, stopping training...".format(patience))

    log.info("Finished training. Best valid loss: {:.6f}".format(best_valid))

if __name__ == "__main__":
    main()
