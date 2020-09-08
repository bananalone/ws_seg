import torch
from torch.utils.tensorboard.writer import SummaryWriter

import argparse

from factories.ClsLabelFactory import ClsLabelFactory
from factories.ClsFactory import ClsFactory
from utils import get_correct_num


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--root", type=str, required=True, help="path to test dataset")
    parse.add_argument("--num_classes", type=int, default=11, help="the total number of classes")
    parse.add_argument("--batch_size", type=int, default=16, help="test batch size")

    return parse.parse_args()


def test(test_loader, model, writer=None):
    model.eval()
    correct_sum = 0
    sum = 0

    for i, batch in enumerate(test_loader, start=1):
        print("====== epoch {i} ======".format(i=i))
        images, target = batch
        outputs = model(images)
        correct = get_correct_num(outputs.detach().numpy(), target.detach().numpy())
        batch_len = images.size(0)
        acc = correct / batch_len
        if writer is not None:
            writer.add_scalar("accuracy for batch", acc, i)

        correct_sum += correct
        sum += batch_len
        print("accuracy: {acc}".format(acc=acc))

    return correct_sum / sum


def main():
    para = "./pretrained/cls_image/ClsResNet_20.pth"
    args = get_args()

    # 工厂
    # factory = ClsLabelFactory(type="test")
    factory = ClsFactory(type="test")
    dataset = factory.get_dataset(root=args.root)
    dataloader = factory.get_dataloader(dataset=dataset, batch_size=args.batch_size)
    model = factory.get_model(num_classes=args.num_classes)
    model.load_state_dict(torch.load(para))

    writer = SummaryWriter()

    with torch.no_grad():
        acc = test(dataloader, model, writer=writer)

    print("total accuracy: {acc}".format(acc=acc))


if __name__ == "__main__":
    main()
