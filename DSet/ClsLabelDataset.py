import os
from torch.utils import data
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


class ClsLabelDataset(data.Dataset):

    def __init__(self, root, imagesize=256, split=500, type="train"):
        super(ClsLabelDataset, self).__init__()
        assert type == "train" or type == "test"
        assert os.path.isdir(root)

        self.transform = transforms.Compose([
            transforms.Resize((imagesize, imagesize)),
            transforms.ToTensor()])

        self.split = split
        self.type = type

        self.data = []  # [(image_path, label)]

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
        image = self.transform(image)
        image[image > 0] = 1
        image = image[0, :, :]
        image = image.view(1, image.size(0), image.size(1))
        return image, label

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
    dataset = ClsLabelDataset(root="../dataset/cls_label")
    labels = dataset.get_labels()
    image, label = dataset[500]
    image = np.transpose(image, axes=(1, 2, 0))
    plt.imshow(image)

    print(image.max())
    print(image.shape)
    print(labels)
    print("label index: {}".format(label))
    print("label: {}".format(labels[label]))
    print("len: {}".format(len(dataset)))

    plt.show()
