from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def img_show(image):
    pil_img = Image.fromarray(np.uint8(image))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_test[0:1]

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
print(img.shape)  # (28, 28)

img_show(img)
