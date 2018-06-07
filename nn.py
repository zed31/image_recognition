#!/usr/local/bin/python3

import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata

DATA_PATH = 'data'

def sigmoid(x):
    """
        Compute the value of the sigmoid at a specific point `x`
        :param: x the point to get the sigmoid value
    """
    return 1.0 / (1.0 + np.exp(-x))

def sigmoidDerivated(x):
    """
        get the derivated of the sigmoid function
        :param: x the derivated at the point x
    """
    return sigmoid(x) * (1 - sigmoid(x))

def toOneHot(y, k):
    """Convert integer to "one-hot" vector.

    to_one_hot(5, 10) -> (0, 0, 0, 0, 1, 0, 0, 0, 0)

    """
    one_hot = np.zeros(k)
    one_hot[y] = 1
    return one_hot

def load_mnist_data():
    """Download and setup MNIST base"""

    mnist = fetch_mldata('MNIST original', data_home=DATA_PATH)

    #Shuffle mnist data
    X, y = shuffle(mnist.data, mnist.target)

    #X is 70 000 x 784 matrices, where each col is an image and each
    #row is a pixel of this image, y is a label for each 70 000 images
    #We divide by 255.0 the X matrix to get the value of each pixels between
    #0 and 1
    return X / 255.0, y

class Layer:
    """A simple neuron layer with `size` neurons and `input_size` inputs"""

    def __init__(self, size, input_size):
        """
            Initialize the layer class with `size` neuron and `input_size` inputs
            :param: size the size of the layer (number of neurons in it)
            :param: input_size the number of neuron in the previous layer
        """

        self.size = size
        self.inputSize = input_size

        #Weight are m*n matrix shuffled m = Neuron number n = Neuron in the previous layer
        self.weightMatrix = np.random.randn(size, self.inputSize)

        #Biases are matrix of size m*1 (also known as a vector)
        self.biasesMatrix = np.random.randn(size)

    def preActivate(self, dataVector):
        """
            Make an aggregation with `dataVector` which is a vector of size `inputSize` containing
            all the output from the previous layer, which is the input of the current layer

            :return: A matrix which contain the pre-activation of each neuron in the layer
        """
        return np.dot(self.weightMatrix, dataVector) + self.biasesMatrix

    def activation(self, output):
        """
            Determine the activation of each layer's neuron by calling the sigmoid function to
            scale the pre activation
        """
        return sigmoid(output)
    
    def derivateActivation(self, output):
        """
            Call the derivate of the activation function thanks to the sigmoid derivate
            :param: output the output of the function
        """
        return sigmoidDerivated(output)
    
    def updateWeight(self, gradient, learningRate):
        """
            Update the weight of the layer thanks to the gradient and learning rate
            :param: gradient the gradient of the current output
            :param: learningRate the user-defined learning rate
        """
        self.weightMatrix -= learningRate * gradient
    
    def updateBiases(self, gradient, learningRate):
        """
            Update the biases of the layer thanks to the gradient and the learning rate
            :param: gradient the gradient of the output for the cost function
            :param: learningRate the user-defined learning rate
        """
        self.biasesMatrix -= learningRate * gradient

    def feedForward(self, dataVector):
        """
            Make the feed forward algorithm of the layer, we take `dataVector` which is a vector
            of `inputSize` size containing the result of the previous layer

            :param: dataVector Vector of inputSize containing activation result from the previous layer
            :return: the activation matrix
        """
        preActivation = self.preActivate(dataVector)
        activationMatrix = self.activation(preActivation)
        return activationMatrix

class Network:
    def __init__(self, input_size):
        """
            Setup the Network class with the `input_size` size

            :param: input_size the number of input in the network 
        """
        self.inputNumber = input_size
        self.layers = []
    
    def addLayer(self, neuron_number):
        """
            Add a new layer with `neuron_number` neurons with the number of
            inputs depending on the previous layer inputs

            :param: neuron_number the number of neurons inside the new layer
        """
        if (len(self.layers) > 0):
            layerInputNumber = self.layers[-1].size
        else:
            layerInputNumber = self.inputNumber
        self.layers.append(Layer(neuron_number, layerInputNumber))

    def feedForward(self, input_data):
        """
            Make the feed forward with the input_data which is a 
            matrix of input
            :param: input_data a matrix containing all the data
        """
        activationMatrix = input_data
        for layer in self.layers:
            activationMatrix = layer.feedForward(activationMatrix)
        return activationMatrix
    
    def predict(self, input_matrix):
        """
            Make the prediction by taking the input matrix and returning the 
            most activated output neuron
            :param: input_matrix the matrix of input data
        """
        return np.argmax(self.feedForward(input_matrix))

    def evaluate(self, X, Y):
        """
            Evaluate all the X matrix (which contains the image and all the pixel related to them)
            then make the percentage of accuracy
        """
        results = [1 if self.predict(x) == y else 0 for (x, y) in zip(X, Y)]
        accuracy = sum(results) / len(results)
        return accuracy

    def train(self, X, Y, steps=30, learning_rate=0.3, batch_size=10):
        """
            Train the network thanks to the input matrix X and the input label Y
            the network will make N `steps`, each with `batch_size` iterations with
            a learning rate of `learning_rate`
        """
        n = Y.size
        for i in range(steps):
            # Just shuffle the datas
            X, Y = shuffle(X, Y)
            for batchStart in range(0, n, batch_size):
                X_batch, Y_batch = X[batchStart:batchStart + batch_size], Y[batchStart:batchStart + batch_size]
                self.trainBatch(X_batch, Y_batch, learning_rate)

    def trainBatch(self, X, Y, learning_rate):
        """
            Make a train batch on X matrix (which contains the image and all the pixels related to them)
            and then apply the learning rate with backpropagation to every layers
        """
        #initialize the gradients for each weights and biases
        weightGradient = [np.zeros(layer.weightMatrix.shape) for layer in self.layers]
        biasGradient = [np.zeros(layer.biasesMatrix.shape) for layer in self.layers]

        #For every images and labels
        for (x, y) in zip(X, Y):
            newWeightGradientMatrix, newBiasesGradientMatrix = self.backPropagation(x, y)
            #We are zipping the new gradient with the previous one to get the new gradient for weights
            weightGradient = [wg + nwg for wg, nwg in zip(weightGradient, newWeightGradientMatrix)]
            biasGradient = [bg + nbg for bg, nbg in zip(biasGradient, newBiasesGradientMatrix)]
        
        # We are making the average of all computed gradients
        avgWeightGradient = [wg / Y.size for wg in weightGradient]
        avgBiasGradient = [bg / Y.size for bg in biasGradient]

        # We are updating the weight and biases thanks to gradient descent
        for layer, weightGradient, biasGradient in zip(self.layers, avgWeightGradient, avgBiasGradient):
            layer.updateWeight(weightGradient, learning_rate)
            layer.updateBiases(biasGradient, learning_rate)



    def backPropagation(self, x, y):
        """
            Apply back propagation algorithm on x vector and compare with y output
            :param: x the x image
            :param: y the y output
        """
        preActivationMatrix = []
        activation = x
        activationMatrix = [activation]

        """
            Compute the matrix of activations and pre activations value to then use
            the derivative to them to propagate the data
        """
        for layer in self.layers:
            preActivation = layer.preActivate(activation)
            preActivationMatrix.append(preActivation)
            activation = layer.activation(preActivation)
            activationMatrix.append(activation)
        
        """
            We compute our cost function using one-hot vectors
        """
        target = toOneHot(int(y), 10)
        outputDelta = self.getOutputDelta(preActivation, activation, target)
        outputDeltaMatrix = [outputDelta]

        #We are going to use vectorial form of the gradient descent function
        #The following code is pure math and vomitive !!
        nbLayers = len(self.layers)
        for outputLayerIndex in reversed(range(nbLayers - 1)):
            currentLayer = self.layers[outputLayerIndex]
            nextLayer = self.layers[outputLayerIndex + 1]
            derivatedActivation = sigmoidDerivated(preActivationMatrix[outputLayerIndex])
            outputDelta = derivatedActivation * np.dot(nextLayer.weightMatrix.transpose(), outputDelta)
            outputDeltaMatrix.append(outputDelta)
        
        #We reverse the delta list cause we get from the end to the begining
        outputDeltaMatrix = list(reversed(outputDeltaMatrix))
        
        #We generate the gradient of the different functions
        weightGradient = []
        biasesGradient = []
        for l in range(len(self.layers)):
            previousActivationValue = activationMatrix[l]
            weightGradient.append(np.outer(outputDeltaMatrix[l], previousActivationValue))
            biasesGradient.append(outputDeltaMatrix[l])

        return weightGradient, biasesGradient


        

    def getOutputDelta(self, z, a, target):
        """
            Get the delta of the target (which is the difference between expected values and output values)
        """
        return a - target



if __name__ == '__main__':
    X, Y = load_mnist_data()
    X_train, Y_train = X[:60000], Y[:60000]
    X_test, Y_test = X[60000:], Y[60000:]

    net = Network(input_size=784)
    net.addLayer(neuron_number=200)
    net.addLayer(neuron_number=10)

    accuracy = net.evaluate(X_test, Y_test)
    print('Initial performance : {:.2f}%'.format(accuracy * 100.0))
    for i in range(30):
        net.train(X_train, Y_train, steps=1, learning_rate=3.0)
        accuracy = net.evaluate(X_test, Y_test)
        print('New performance : {:.2f}%'.format(accuracy * 100.0))
