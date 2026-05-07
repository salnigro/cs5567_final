import os
from collections import defaultdict

def parse_gt_file(file_path):

    data = defaultdict(list)

    with open(file_path, "r") as f:
        for line in f:
            frame, obj_id, x, y, w, h, conf, cls, vis = line.strip().split(",")

            frame = int(frame)
            obj_id = int(obj_id)

            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)

            bbox = [x, y, x + w, y + h]

            data[frame].append({
                "id": obj_id,
                "bbox": bbox
            })

    return data
from torchvision import transforms

color_aug = transforms.Compose([
    transforms.ColorJitter(
        brightness=0.5,
        contrast=0.5,
        saturation=0.5
    ),
    transforms.GaussianBlur(
        kernel_size=(5,9),
        sigma=(0.1,5)
    )
])

import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MOTDataset(Dataset):

    def __init__(self, root_dir, sequence, transform=None):

        self.root_dir = root_dir
        self.sequence = sequence
        self.transform = transform

        self.img_dir = os.path.join(root_dir, sequence, "img1")
        gt_path = os.path.join(root_dir, sequence, "gt", "gt.txt")

        self.gt_data = parse_gt_file(gt_path)
        self.frames = sorted(self.gt_data.keys())

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):

        frame = self.frames[idx]

        img_path = os.path.join(
            self.img_dir,
            f"{frame:06d}.jpg"
        )

        image = Image.open(img_path).convert("RGB")

        objects = self.gt_data[frame]

        boxes = []
        labels = []

        for obj in objects:
            boxes.append(obj["bbox"])
            labels.append(1)   # pedestrian class

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transform:
            image = self.transform(image)

        image = self.to_tensor(image)

        return image, target

        
if __name__ == "__main__":

    train_dir = r"c:\Users\salni\Documents\cs5567\train"

    all_train_data = {}

    for sequence in os.listdir(train_dir):

        seq_path = os.path.join(train_dir, sequence)
        gt_path = os.path.join(seq_path, "gt", "gt.txt")

        if os.path.exists(gt_path):
            gt_data = parse_gt_file(gt_path)
            all_train_data[sequence] = gt_data

    print("Loaded sequences:", all_train_data.keys())
