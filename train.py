from CNN import SimpleConvNet
from dataset.mnist import load_mnist
from trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = SimpleConvNet()

trainer = Trainer(network, x_train, t_train, x_test, t_test)

trainer.train()

network.save_params("params.pkl")

