from factories.AbstractFactory import AbstractFactory

from torch.utils.data.dataloader import DataLoader

from DSet.ClsDataset import ClsDataset
from models.ResNet import ClsResNet


class ClsFactory(AbstractFactory):

    def __init__(self, type="train"):
        super(ClsFactory, self).__init__(type=type)

    def get_dataset(self, root):
        return ClsDataset(root=root, type=self.type)

    def get_dataloader(self, dataset, batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=self.shuffle)

    def get_model(self, num_classes, pretrained=True):
        return ClsResNet(num_classes=num_classes, pretrained=pretrained)