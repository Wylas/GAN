import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

# CREAR EL DATASET

n = 500
p = 2

# INPUT & OUTPUT
X, Y = make_circles(n_samples=n, factor=0.4, noise=0.05)

plt.scatter(X[:, 0], X[:, 1])
plt.axis("equal")
plt.show()

# CLASE DE LA CAPA DE LA RED

class neural_layer():
    def __init__(self, n_conn, n_neur, act_f):
        self.act = act_f
        self.b = np.random.rand(1, n_neur) * 2 - 1
        self.w = np.random.rand(n_conn, n_neur) * 2 - 1


# FUNCIONES DE ACTIVACIÓN


#definimos la función sigmoide

def sigm(x):
    return 1 / (1 + np.e ** (-x))


_X = np.linspace(-5, 5, 100)
plt.plot(_X, sigm(_X))
plt.show()



# CREAMOS LAS CAPAS

#l0 = neural_layer(p, 4, sigm)
#l1 = neural_layer(4, 8, sigm)

#numero de neuronas que va a tener cada capa, el output siempre tiene que ser 1
topology = [p, 4, 8, 16, 8, 4, 1]

def create_nn(topology, act_f):
    nn = []
    for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[l], topology[l + 1], act_f))
        print(nn)

    return nn


neural_net = create_nn(topology, sigm)



#TRAIN

def l2_cost(Vp, Vr):
    return np.mean((Vp - Vr) ** 2)


def train(neural_net, X, Y, l2_cost, lr=0.5):

    #Forward pass

    out = [(None, X)]

    for l, layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].w + neural_net[l].b
        a = neural_net[l].act_f[1](z)

        out.append((z, a))

    print(out[-1][1])


train(neural_net, X, Y, l2_cost, 0.5)



