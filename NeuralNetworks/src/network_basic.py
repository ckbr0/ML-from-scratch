from random import shuffle
import numpy as np

class Network:

    def __init__(self, layers, activation_function, activation_function_prime, cost_function_prime):
        self.n_layers = len(layers)
        self.layers = layers
        self.biases = [np.random.randn(y) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.activation_function = activation_function
        self.activation_function_prime = activation_function_prime
        self.cost_function_prime = cost_function_prime

    """"""
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.activation_function(np.dot(w, a) + b)
        return a

    """ Streniraj nevronsko mrežo """
    def stochastic_gradient_descent(self, training_data, batch_size, eta):
        n = len(training_data)
        shuffle(training_data)
        # Razdeli podatke za treniranje v manjše skupke.
        batches = [training_data[i:i+batch_size] for i in range(0, n, batch_size)]
        # Na vsakem skupku izvedi gradientni spust.
        for batch in batches:
            self.gradient_descent(batch, eta)

    """ Gradientni spust """
    def gradient_descent(self, training_data, eta):
        n = len(training_data)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in training_data:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b-(eta/n)*nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w-(eta/n)*nw for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feed forward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_function_prime(activations[-1], y) * self.activation_function_prime(zs[-1])
        #print(activations[-1].shape,"\n", y.shape)
        nabla_b[-1] = delta
        nabla_w[-1] = np.outer(delta, activations[-2])
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.n_layers):
            z = zs[-l]
            sp = self.activation_function_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.outer(delta, activations[-l-1])
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

