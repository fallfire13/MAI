from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import time

def sigmoid(x, derivative=False):
    if derivative:
        return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
    return 1 / (1 + np.exp(-x))


def softmax(x, derivative=False):
    # Numerically stable with large exponentials
    exps = np.exp(x - x.max())
    if derivative:
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)

def tanh(x, derivative=False):
    if derivative:
        return 1 - (np.tanh(x))**2
    return np.tanh(x)

class NN():
    def __init__(self, sizes, activations, epochs=10, l_rate=0.001):
        self.sizes = sizes
        self.activations = activations
        self.epochs = epochs
        self.l_rate = l_rate

        # we save all parameters in the neural network in this dictionary
        self.params = self.initialization()

    def initialization(self):
        # number of nodes in each layer
        if len(self.sizes) == 1:
            return
        if len(self.sizes) == 2:
            input_layer = self.sizes[0]
            output_layer = self.sizes[1]
            params = {'W1': np.random.randn(output_layer, input_layer) * np.sqrt(1. / output_layer), }
        else:
            input_layer=self.sizes[0]
            hidden = [self.sizes[1]]
            params = {
                'W1': np.random.randn(hidden[0], input_layer) * np.sqrt(1. / hidden[0])
            }
            for i in range(2, len(self.sizes)-1):
                hidden.append(self.sizes[i-1])
                w = 'W{}'.format(i)
                params[w] = np.random.randn(hidden[i-1], hidden[i-2]) * np.sqrt(1. / hidden[i-1])
            output_layer=self.sizes[-1]
            w = 'W{}'.format(len(self.sizes)-1)
            params[w] = np.random.randn(output_layer, hidden[-1]) * np.sqrt(1. / output_layer)

        return params

    def forward_pass(self, x_train):
        params = self.params

        params['A0'] = x_train

        # input layer to hidden layer 1
        for i in range(1, len(self.sizes)):
            params['Z{}'.format(i)] = np.dot(params["W{}".format(i)], params['A{}'.format(i-1)])
            params['A{}'.format(i)] = self.activations[i-1](params['Z{}'.format(i)])

        return params['A{}'.format(i)]

    def backward_pass(self, y_train, output):
        params = self.params
        change_w = {}

        error = 2 * (output - y_train) / output.shape[0] * softmax(params['Z{}'.format(len(self.sizes)-1)], derivative=True)
        change_w['W{}'.format(len(self.sizes)-1)] = np.outer(error, params['A{}'.format(len(self.sizes)-2)])
        for i in reversed(range(2, len(self.sizes))):
            error = np.dot(params['W{}'.format(i)].T, error) * self.activations[i-1](params['Z{}'.format(i-1)], derivative=True)
            change_w['W{}'.format(i-1)] = np.outer(error, params['A{}'.format(i-2)])

        return change_w

    def update_network_parameters(self, changes_to_w):
        
        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def compute_accuracy(self, x_val, y_val):
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
        
        return np.mean(predictions)

    def predict(self, x):
        return np.argmax(self.forward_pass(x),0)

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        accs = []
        for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)

            
            accuracy = self.compute_accuracy(x_val, y_val)
            accs.append(accuracy)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, accuracy * 100
            ))
        self.history = accs
