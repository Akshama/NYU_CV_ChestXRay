import csv
import torch

from pathlib import Path
from PIL import Image
from torchvision import transforms


NUM_CLASSES = 14

DATA_LABELS = {
    0: "Path",
    1: "Sex",
    2: "Age",
    3: "Frontal/Lateral",
    4: "AP/PA", # - no idea what this is
    5: "No Finding",
    6: "Enlarged Cardiomediastinum",
    7: "Cardiomegaly",
    8: "Lung Opacity",
    9: "Lung Lesion",
    10: "Edema",
    11: "Consolidation",
    12: "Pneumonia",
    13: "Atelectasis",
    14: "Pneumothorax",
    15: "Pleural Effusion",
    16: "Pleural Other",
    17: "Fracture",
    18: "Support Devices",
}


def chexpert_collate(batch):
    xs = [ex[0] for ex in batch]
    
    need_resize = False
    min_width = None
    min_height = None
    for x in xs:
        w, h = x.size(1), x.size(2)
        if min_width == None:
            min_width = w
            min_height = h
        else:
            if w != min_width:
                min_width = min(w, min_width)
                need_resize = True
            if h != min_height:
                min_height = min(h, min_height)
                need_resize = True
    
    if need_resize:
        for x in xs:
            x.resize_(x.size(0), min_width, min_height)
    
    xs = torch.stack(xs, 0).expand(len(xs), 3, min_width, min_height)

    ys = [ex[1] for ex in batch]
    y_dim = len(ys[0])
    ys = [torch.stack([y[i] for y in ys], 0) for i in range(y_dim)]
    
    return (xs, ys)


PATH_MAP = {
    "": -2,
    "-1.0": -1,
    "0.0": 0,
    "1.0": 1,
}


class ChexpertDataset(torch.utils.data.Dataset):
    def __init__(self, path="/scratch/ax2028/CheXpert-dataset/", train=True):
        super().__init__()

        self.base_path = Path(path)
        if not self.base_path.exists():
            raise FileNotFoundError(self.base_path)

        self.exs = {}
        self.metadata = []
        self.size = -1
        csvpath = self.base_path / ("train.csv" if train else "valid.csv")
        with open(csvpath, newline='') as read:
            reader = csv.reader(read)
            for row in reader:
                if self.size != -1:  # skip header
                    self.metadata.append(row)
                self.size += 1
        
        self.to_tensor = transforms.ToTensor()
       
        self.ONE = torch.tensor([1])
        self.ZERO = torch.tensor([0])
        self.UNC = torch.tensor([-1])
            
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # uncached load directly from file
        r = self.metadata[idx]
        pth = self.base_path / Path('/'.join(Path(r[0]).parts[1:]))
        sex = self.ONE if r[1] == "Female" else self.ZERO
        age = torch.Tensor([int(r[2]) / 120])  # normalize by lifespan
        angle = self.ONE if r[3] == "Frontal" else self.ZERO
        pathologies = torch.Tensor([PATH_MAP[r[i]] for i in range(5, len(r))])
        img = self.to_tensor(Image.open(pth))
        return (img, (sex, age, angle, pathologies))
