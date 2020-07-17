from CNN import SimpleConvNet
from dataset.mnist import load_mnist

network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

network.load_params("params.pkl")

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

print(network.predict(x_test[0:1]).argmax(axis=1))
