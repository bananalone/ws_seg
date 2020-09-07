class AbstractFactory:

    def __init__(self, type="train"):
        assert type == "train" or type == "test"
        self.type = type
        self.shuffle = True if self.type == "train" else False

    def get_dataset(self, root):
        raise NotImplementedError("get_dataset is not implemented!")

    def get_dataloader(self, dataset, batch_size):
        raise NotImplementedError("get_dataloader is not implemented!")

    def get_model(self, num_classes, pretrained=True):
        raise NotImplementedError("get_model is not implemented!")