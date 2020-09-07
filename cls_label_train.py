from torch import nn
from torch import optim

from torch.utils.tensorboard import SummaryWriter


from utils import adjust_lr, get_correct_num, save_model
from factories.ClsLabelFactory import ClsLabelFactory
from factories.ClsFactory import ClsFactory

import argparse


def get_args():
    '''
    训练用的一些基本设置
    :return:
    '''
    parse = argparse.ArgumentParser()

    parse.add_argument("--root", type=str, required=True, help="path to dataset")
    parse.add_argument("--num_classes", type=int, default=11, help="the total number of classes")
    parse.add_argument("--batch_size", type=int, default=16, help="batch size for dataloader")
    parse.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    parse.add_argument("--num_epoches", type=int, default=90, help="the total number of training")
    parse.add_argument("--device", type=str, default="cpu", help="use cuda or cpu, default cpu")

    return parse.parse_args()


def train(train_loader, model, criterion, optimizer, epoch, args, writer=None):
    '''
    训练函数
    :param train_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param epoch:
    :param args:
    :param writer:
    :return:
    '''
    model.train()

    sum_correct = 0 # 总的正确个数
    sum = 0 # 总的个数
    for i, batch in enumerate(train_loader, start=1):
        images, target = batch
        images.to(args.device)
        target.to(args.device)

        # 向前传播
        outputs = model(images)

        loss = criterion(outputs, target)

        # 调整学习率
        adjust_lr(optimizer, epoch, args)

        # 向后传播, 优化参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录当前batch正确率
        correct = get_correct_num(outputs.detach().numpy(), target.detach().numpy())
        acc = correct / target.size(0)

        # 记录总正确率
        sum_correct += correct
        sum += target.size(0)

        print("### {i}/{length} : loss {loss}   acc {acc}".format(i=i, length=len(train_loader), loss=loss, acc=acc))

        # 可视化
        if writer is not None:
            step = (epoch - 1) * len(train_loader) + i
            writer.add_scalar("Loss/train",  loss, global_step=step)
            writer.add_scalar("Accuracy/train", sum_correct/sum, global_step=step)


def main():
    '''
    main
    :return:
    '''
    # 获得训练用的超参数
    args = get_args()

    # 工厂
    factory = ClsLabelFactory(type="train")
    dataset = factory.get_dataset(root=args.root)
    dataloader = factory.get_dataloader(dataset=dataset, batch_size=args.batch_size)
    model = factory.get_model(num_classes=args.num_classes, pretrained=True)
    model.to(args.device)

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    criterion.to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # 可视化
    writer = SummaryWriter()

    # 开始训练
    for i in range(args.num_epoches):
        epoch = i + 1
        print("===== epoch: {epoch} =====".format(epoch=epoch))
        train(dataloader, model, criterion, optimizer, epoch, args, writer=writer)
        if epoch % 10 == 0:
            save_model(model, "./pretrained/cls_label", name="ClsLabelResNet_{epoch}.pth".format(epoch=epoch))


if __name__ == "__main__":
    main()