import os
from torch.utils import data
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


class ClsDataset(data.Dataset):

    def __init__(self, root, imagesize = 256, split = 500, type = "train"):
        super(ClsDataset, self).__init__()
        assert type == "train" or type == "test"
        assert os.path.isdir(root)

        self.transform = transforms.Compose([
            transforms.Resize((imagesize, imagesize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.split = split
        self.type = type

        self.data = [] # [(image_path, label)]

        self.labels = os.listdir(root)
        self.labels = [label for label in self.labels if os.path.isdir(os.path.join(root, label))]
        self.labels.sort()
        for idx, label in enumerate(self.labels, start=0):
            label_path = os.path.join(root, label)
            self._extend_class(label_path, idx)


    def __getitem__(self, item):
        image_path, label = self.data[item]
        image = Image.open(image_path)
        image.convert("RGB")
        return self.transform(image), label


    def __len__(self):
        return len(self.data)


    def _extend_class(self, label_path, label_idx):
        images = [(os.path.join(label_path, img), label_idx) for img in os.listdir(label_path) \
                  if img.endswith(".jpg") or img.endswith(".png")]
        images.sort()
        if self.type == "train":
            images = images[0:self.split]

        elif self.type == "test":
            images = images[self.split:]

        self.data.extend(images)


    def get_labels(self):
        return self.labels


if __name__ == "__main__":
    dataset = ClsDataset(root="../dataset/cls")
    labels = dataset.get_labels()
    image, label = dataset[500]
    image = np.transpose(image, axes=(1, 2, 0))
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    plt.imshow(image)
    print(labels)
    print("label index: {}".format(label))
    print("label: {}".format(labels[label]))
    print("len: {}".format(len(dataset)))

    plt.show()
