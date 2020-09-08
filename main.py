from CAM.GradCAM import GradCam, cam2img
from factories.ClsFactory import ClsFactory
from factories.ClsLabelFactory import ClsLabelFactory

import torch
from torchvision.models.resnet import resnet50
from matplotlib import pyplot as plt
import cv2
import numpy as np

def main():
    """

    :return:
    """
    model_path = "./pretrained/cls_label/ClsLabelResNet_20.pth"
    image_path = "./dataset/cls_label"
    img_idx = 800
    ship = 510
    factory = ClsLabelFactory(type="test")
    model = factory.get_model(num_classes=11)
    model.load_state_dict(torch.load(model_path))
    dataset = factory.get_dataset(root=image_path)
    print(dataset.data[img_idx])
    ipt, target = dataset[img_idx]
    image = np.transpose(ipt, axes=(1, 2, 0))
    ipt = ipt.view(1, 1, 256, 256)
    plt.imshow(image)
    # plt.show()
    # model = resnet50(pretrained=True)
    grad_cam = GradCam(model=model, layer="layer4")
    cam = grad_cam(ipt, target_class=None)
    # plt.imshow(cam2img(image.detach().numpy(), cam))
    plt.imshow(cam)
    plt.show()



if __name__ == '__main__':
    main()
