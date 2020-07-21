from dataset.mnist import load_mnist
from PIL import Image


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=False)

Image.fromarray(x_test[7].reshape(28, 28)).save("a.png")
