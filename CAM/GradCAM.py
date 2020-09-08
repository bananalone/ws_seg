import torch
import numpy as np
import cv2
from torchvision.models import vgg16
import matplotlib.pyplot as plt


class FeatureExtractor:

    def __init__(self, model, layer):
        self.__model = model
        self.__layer = layer
        self.__grad = None

    def hook_fn(self, grad):
        self.__grad = grad

    def get_grad(self):
        return self.__grad

    def __call__(self, x):
        features = None
        for name, module in self.__model.named_children():
#             print(name)
#             print(x.shape)
            if name in ["fc", "classifier"]:
                x = torch.flatten(x, 1)

            x = module(x)
            if name == self.__layer:
                # 目标层
                features = x
                x.register_hook(self.hook_fn)
        return features, x



class GradCam:

    def __init__(self, model, layer):
        self.__model = model
        self.__feature_extractor = FeatureExtractor(self.__model, layer)

    def __call__(self, x, target_class=None):
#         print(x.shape)
        self.__model.eval()
        self.__model.zero_grad()
        features, y = self.__feature_extractor(x)
        cls = target_class if target_class is not None else \
                np.argmax(y.detach().numpy(), axis=1)[0]
#         print(cls)
        res = y[0, cls]
        res.backward(retain_graph=True)

        grad = self.__feature_extractor.get_grad()[0].detach().numpy()
        features = features[0].detach().numpy()

#         print(features.shape)
#         print(grad.shape)

        cam = np.zeros(features.shape[1:], dtype=np.float)
#         print(cam.shape)

        for i in range(features.shape[0]):
            w = np.mean(grad[i])
            feature = features[i]
            feature[feature < 0] = 0 # relu
            # cam += w * feature
            cam += feature * grad[i]

#         print(cam.shape)
        cam[cam < 0] = 0 # relu
#         print(cam.min())
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (256, 256))


#         cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
#         cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

        return cam



def cam2img(img, cam):
    img = img * 255
    img = img.astype(np.uint8)
    cam = cam * 255
    cam = cam.astype(np.uint8)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    res = np.empty(img.shape)
    for i in range(3):
        res[:,:,i] = img[:,:,i] * 0.7 + cam[:,:,i] * 0.3

    res = res.astype(np.int)
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    return res




