import torch
from torch import nn
import math
import matplotlib.pyplot as plt

def generate_data(num_samples, num_in_features, num_out_features):
    """
    Function generates x,y pairs for training
        - x shape is N (samples) x F (in_features)
        - y shape is N (samples) x D (out_features)
    """
    # sample N inputs from standard multivariate gaussian
    samples = torch.randn((num_in_features, num_samples))

    # build ground-truth model
    W_1 = math.sqrt(2 / num_in_features) * torch.randn((64, num_in_features))
    b_1 = math.sqrt(2 / num_in_features) * torch.randn((64, 1))
    W_2 = math.sqrt(2 / num_in_features) * torch.randn((num_out_features, 64))
    b_2 = math.sqrt(2 / num_in_features) * torch.randn((num_out_features, 1))
    #W_3 = torch.randn((num_out_features, 32))
    #b_3 = torch.randn((num_out_features, 1))

    # generate y data
    nl = nn.ReLU()
    x_1 = nl(torch.matmul(W_1, samples) + b_1)
    x_2 = nl(torch.matmul(W_2, x_1) + b_2)
    #x_3 = nl(torch.matmul(W_3, x_2) + b_3)

    return samples, x_2

def split(x, y, train_p, val_p, test_p):
    """
    Function splits data into train, val, test
    - x is (num_in_features, num_samples)
    - y is (num_out_features, num_samples)
    - train_p + val_p + test_p = 1, all >= 0
    """
    n = x.shape[1]
    perm = torch.randperm(n)
    train_stop = int(train_p * n)
    val_stop = int((train_p + val_p) * n)
    train_indices = perm[:train_stop]
    val_indices = perm[train_stop:val_stop]
    test_indices = perm[val_stop:]
    xtrain, ytrain = x[:, train_indices], y[:, train_indices]
    xval, yval = x[:, val_indices], y[:, val_indices]
    xtest, ytest = x[:, test_indices], y[:, test_indices]
    return xtrain, ytrain, xval, yval, xtest, ytest

class LinearLayer():
    """
    Neural Network Linear Layer
    """

    def __init__(self, in_features, out_features, bias=True):

        # initialize parameters of neural network layer
        self.initialize_parameters(in_features, out_features, bias)

    def initialize_parameters(self, in_features, out_features, bias):
        """
        Function to initialize parameters
        """
        
        # initialize weights of nn layer
        self.weight = math.sqrt(2/in_features) * torch.randn((out_features, in_features))

        # if bias requested, initialize
        if bias:
            self.bias = math.sqrt(2/in_features) * torch.randn((out_features, 1))

    def forward(self, x):
        """
        Forward to compute output of layer given layer input
        - x is shape (in_features x 1)
        """

        # multiply input by weights matrix, add bias, and return
        return torch.matmul(self.weight, x) + self.bias
    
    def backward(self, x, grad_wrt_out):
        """
        Function to compute gradient of loss wrt parameters and input given input and grad loss wrt forward output
        - grad_wrt_out is of shape out.shape (out_features x 1)
        - x is of shape (in_features x 1)
        - compute gradient of loss wrt parameters and perform parameter update
        - compute gradient of loss wrt input and return
        """

        # compute gradient of loss wrt weights matrix given grad loss wrt layer output and input values
        grad_w = torch.matmul(grad_wrt_out, x.T)

        # propagate gradient of loss wrt bias
        grad_b = grad_wrt_out
        
        # compute gradient of loss wrt layer inputs
        grad_x = torch.matmul(self.weight.T, grad_wrt_out)

        # update weights matrix
        self.weight -= 1e-5 * grad_w

        # update biad
        self.bias -= 1e-5 * grad_b

        # return grad loss wrt layer inputs
        return grad_x
    
class ReLU():

    def __init__(self):
        pass

    def forward(self, x):
        """
        Forward to compute output of layer given layer input
        - x is shape (in_features x 1)
        """

        # zero out all non-zero entries
        f = x.clone()
        f[f < 0] = 0

        return f
    
    def backward(self, grad_wrt_out, x):
        """
        Backward to compute grad_wrt_input (in_features x 1) from grad_wrt_output (out_features x 1)
        """
        mask = x.clone()
        mask[mask < 0] = 0
        return grad_wrt_out * mask


class MSELossLayer():
    def __init__(self):
        pass
    def forward(self, pred, target):
        """
        Forward to compute loss given input, target
        Pred is (out_features x 1), target is (out_features x 1)
        """
        diffs = pred - target
        loss = torch.matmul(diffs.T, diffs)
        return loss.item()
    
    def backward(self, pred, target):
        """
        Backward to compute grad loss wrt model output (out_features x 1) from pred, target
        Pred is (out_features x 1), target is (out_features x 1)
        """
        return 2*(pred - target)

    

# generate some toy data
x, y = generate_data(1000, 16, 4)
#print(x.shape, y.shape)

# split into train, val, test sets
xtrain, ytrain, xval, yval, xtest, ytest = split(x, y, 0.7, 0.15, 0.15)
#print(xtrain.shape, ytrain.shape, xval.shape, yval.shape, xtest.shape, ytest.shape)

# create model
l1 = LinearLayer(16, 64)
l2 = LinearLayer(64, 4)
r = ReLU()
loss = MSELossLayer()

# hold losses
losses = []

# get input data
for e in range(100):
    for i in range(x.shape[1]):
        x_in = x[:, i].unsqueeze(-1)
        y_in = y[:,i].unsqueeze(-1)
        #print(x_in,y_in)

        # compute outputs
        h1 = l1.forward(x_in)
        z1 = r.forward(h1)
        h2 = l2.forward(z1)
        z2 = r.forward(h2)

        # compute loss
        obj = loss.forward(z2, y_in)

        # compute grad loss wrt z2
        grad_loss_z2 = loss.backward(z2, y_in)

        # compute grad loss wrt h2
        grad_loss_h2 = r.backward(grad_loss_z2, h2)

        # compute grad loss wrt z1 and update linear layer
        grad_loss_z1 = l2.backward(z1, grad_loss_h2)

        # compute grad loss wrt h1
        grad_loss_h1 = r.backward(grad_loss_z1, h1)

        # compute grad loss wrt x_in and update linear layer
        grad_loss_x_in = l1.backward(x_in, grad_loss_h1)

    print(obj)
    losses.append(obj)

plt.plot(losses)
plt.show()