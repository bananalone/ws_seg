from factories.AbstractFactory import AbstractFactory

from torch.utils.data.dataloader import DataLoader

from DSet.ClsLabelDataset import ClsLabelDataset
from models.ResNet import ClsLabelResNet

class ClsLabelFactory(AbstractFactory):

    def __init__(self, type="train"):
        super(ClsLabelFactory, self).__init__(type=type)

    def get_dataset(self, root):
        return ClsLabelDataset(root=root, type=self.type)

    def get_dataloader(self, dataset, batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=self.shuffle)

    def get_model(self, num_classes, pretrained=True):
        return ClsLabelResNet(num_classes=num_classes, pretrained=pretrained)