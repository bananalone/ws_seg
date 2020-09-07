import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from DSet.ClsDataset import ClsDataset
from DSet.ClsLabelDataset import ClsLabelDataset

def get_cls_dataloader(root, batch_size, type="train"):
    assert type == "train" or type == "test"
    shuffle = True if type == "train" else False
    loader = DataLoader(ClsDataset(root, type=type), batch_size = batch_size, shuffle = shuffle)
    return loader

def get_cls_label_dataloader(root, batch_size, type="train"):
    assert type == "train" or type == "test"
    shuffle = True if type == "train" else False
    loader = DataLoader(ClsLabelDataset(root, type=type), batch_size = batch_size, shuffle = shuffle)
    return loader


def adjust_lr(optimizer, epoch, args, decay_rate=30):
    lr = args.lr * (0.1 ** (epoch // decay_rate))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_correct_num(outputs, target):
    if outputs.shape[0] == target.shape[0]:
        labels = np.argmax(outputs, axis=1)
        correct = labels == target
        return correct.sum()
    else:
        return 0

def save_model(model, path, name):
    if not os.path.exists(path):
        os.makedirs(path)

    file = os.path.join(path, name)
    torch.save(model.state_dict(), file)


if __name__ == "__main__":
    loader = get_cls_dataloader("./dataset/cls", batch_size=16)
    batch = next(iter(loader))
    images, labels = batch
    print("images shape:", images.size())
    print("labels shape", labels.size())
