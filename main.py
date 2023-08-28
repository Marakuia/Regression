import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Network(nn.Module):
    def __init__(self, num_layers, num_neurons, act_func, batch_size=50, *args, **kwargs):
        """ Constructor parameters:
        :param num_layers: Number of hidden layers in the network
        :param num_neurons: Number of neurons in the layers
        :param act_func: Activation function for layers
        :param batch_size: Data packet size
        """
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.act_func = act_func
        # network skeleton
        self.input_layer = nn.Linear(batch_size, self.num_neurons)
        self.hidden_layer = nn.Linear(self.num_neurons, self.num_neurons)
        self.output_layer = nn.Linear(self.num_neurons, batch_size)

    def forward(self, x):
        """ Pass through the layers
        :param x: Input data
        """
        # pass through the layers depending on the number of hidden layers
        x = self.activation_f(self.input_layer(x))
        for i in range(self.num_layers):
            x = self.activation_f(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

    def activation_f(self, x):
        """ Activation function selection
        :param x: Weighted sum
        """
        match self.act_func:
            case "sigmoid":
                return f.sigmoid(x)
            case "tanh":
                return f.tanh(x)
            case "softmax":
                return f.softmax(x)
            case "relu":
                return f.relu(x)


def fit(model, opt, los_fn, data, validation, epochs=250):
    """ Neural network training

    :param model: Neural network class object
    :param opt: Selected optimizer
    :param los_fn: Selected activation function
    :param data: Training Data
    :param validation: Validation Data
    :param epochs: Number of epochs
    """
    plt.figure(figsize=(50, 16))
    training_losses = []
    val_losses = []
    for epoch in range(epochs):
        training_loss = 0.0
        validation_loss = 0.0
        validation_result = []

        model.train()
        for batch in data:
            opt.zero_grad()
            inputs, target = batch[:, 0], batch[:, 1]
            inputs, target = Variable(inputs), Variable(target)
            output = model(inputs)
            loss = los_fn(output, target)
            loss.backward()
            opt.step()

            training_loss += loss.data.item()
        training_loss /= len(data)
        training_losses.append(training_loss)

        model.eval()
        for batch_v in validation:
            inputs_v, target_v = batch_v[:, 0], batch_v[:, 1]
            output_v = model(inputs_v.double())
            loss = los_fn(output_v, target_v.double())
            validation_loss += loss.data.item()
            validation_result.append(output_v.detach().numpy())

        validation_loss /= len(validation)
        val_losses.append(validation_loss)
        if epoch % 10 == 0:
            print('Train Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}'.format(epoch, training_loss,
                                                                                           validation_loss))
        if epoch % 30 == 0:
            plt.subplot(4, 3, int(epoch / 30) + 1)
            plt.scatter(val_data[:, 0], val_data[:, 1])
            plt.scatter(val_data[:, 0], np.array(validation_result).reshape(200))
            plt.title(epoch)
    return training_losses, val_losses


def predict(model, data):
    """ Validation Data Predictions
    :param model: Neural network class object
    :param data: Validation Data
    :return: Predictions in list form
    """
    result = []
    for batch in data:
        inputs, target = batch[:, 0], batch[:, 1]

        output = model(inputs.double())

        result.append(output.detach().numpy())
    return result


# dataset preparation
train_data = pd.read_csv('/home/kseniia/PycharmProjects/Regression/Practice/Regression/train_dataset', sep=',')
val_data = pd.read_csv('/home/kseniia/PycharmProjects/Regression/Practice/Regression/validation_dataset', sep=',')

train_data = train_data.to_numpy()
train_loader = DataLoader(train_data, batch_size=50)
val_data = val_data.to_numpy()
val_loader = DataLoader(val_data, batch_size=50)
net = Network(1, 100, 'relu').double()
learning_rate = 0.001
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

train_losses, validation_losses = fit(net, optimizer, criterion, train_loader, val_loader)

pred = predict(net, val_loader)
pred = np.array(pred).reshape(200)
plt.subplot(4, 3, 11)
plt.scatter(train_data[:, 0], train_data[:, 1])
plt.scatter(train_data[:, 0], pred)
plt.title("Result")
plt.show()

plt.figure(figsize=(23, 16))
plt.plot(np.arange(250), np.array(train_losses), label="Train losses")
plt.plot(np.arange(250), np.array(validation_losses), label="Validation losses")
plt.legend()
plt.title("MSE")
plt.show()
