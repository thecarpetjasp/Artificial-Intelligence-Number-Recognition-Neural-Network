import numpy as np
import os
from PIL import Image
import json

class Layer_Dense:
    def __init__(self, n_inputs, n_outputs):
        self.weights = 0.01 * np.random.randn(n_inputs, n_outputs)
        self.biases = np.zeros((1, n_outputs))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
    def save(self):
        return (self.weights.copy(), self.biases.copy())

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self, dvalues, y):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y.shape) == 1:
            y = np.eye(labels)[y]
        self.dinputs = y / dvalues
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    def forward(self, inputs, y):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y)
    def backward(self, dvalues, y):
        samples = len(dvalues)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y] -= 1
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay=0, momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.biases_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.biases_momentums - self.current_learning_rate * layer.dbiases
            layer.biases_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates
    def post_update_params(self):
        self.iterations += 1


# Initialise ######################################################################################################
        
os.chdir('./') # Change to directory that contain test data images

# Import samples into program
X = []
y = []
for img in [x for x in os.listdir() if '.png' in x]:
    image = np.array(Image.open(img).convert('L')).reshape(784) / 255.0
    X.append(image.tolist())
    y.append(int(img[0]))
X = np.array(X)
y = np.array(y)

# Creating neural network layers and parameters
dense1 = Layer_Dense(784, 128)
dense2 = Layer_Dense(128, 10)
a1 = Activation_ReLU()
a2 = Activation_Softmax()
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(decay=1e-1, momentum=0.9)
accuracy = 0.0
lowest_score = 999999


# Training ########################################################################################################

try:
    for epoch in range(100000):
        print(f"Epochs: {epoch}, Loss: {lowest_score}, Accuracy: {accuracy*100:.2f}%", end="\r", flush=True)
        dense1.forward(X)
        a1.forward(dense1.output)
        dense2.forward(a1.output)
        a2.forward(dense2.output)
        loss = loss_activation.forward(dense2.output, y)
        if loss < lowest_score:
            predictions = np.argmax(a2.output, axis=1)==y
            accuracy = np.mean(predictions)
            lowest_score = loss
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        a1.backward(dense2.dinputs)
        dense1.backward(a1.dinputs)
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
    else:
        print(f"Learning: {a}, Score: {lowest_score}, Accuracy: {accuracy*100:.2f}%")
except:
    print(f"Learning: {a}, Score: {lowest_score}, Accuracy: {accuracy*100:.2f}%")
    pass

save = input("Save? (Leave blank if no): ")
if save:
    data = {'dense1' : {'weights' : dense1.weights.tolist(), 'biases' : dense1.biases.tolist()}, 'dense2' : {'weights' : dense2.weights.tolist(), 'biases' : dense2.biases.tolist()}}
    with open('Number_Recognition_Model_Trained.json', 'w') as file:
        file.write(json.dumps(data, indent=4))