import torch
import math

class LinearLayer():
    """
    Neural Network Linear Layer
    """

    def __init__(self, in_features, out_features, bias=True):

        # save params
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        # initialize parameters of neural network layer
        self.initialize_parameters()

        # hold previous values
        self.last_in = None
        self.grad_w = None
        self.grad_b = None

    def initialize_parameters(self):
        """
        Function to initialize parameters
        """
        
        # initialize weights of nn layer
        self.weight = math.sqrt(2/self.in_features) * torch.randn((self.out_features, self.in_features))

        # if bias requested, initialize
        if self.has_bias:
            self.bias = math.sqrt(2/self.in_features) * torch.randn((self.out_features, 1))

    def forward(self, x):
        """
        Forward to compute output of layer given layer input
        - x is shape (in_features x 1)
        """

        # cache input
        self.last_in = x

        # if bias, multiply input by weights matrix, add bias, and return
        if self.has_bias:
            torch.matmul(self.weight, x) + self.bias

        # else multiply input by weights matrix and return
        return torch.matmul(self.weight, x)
    
    def backward(self, grad_wrt_out, x):
        """
        Function to compute gradient of loss wrt parameters and input given input and grad loss wrt forward output
        - grad_wrt_out is of shape out.shape (out_features x 1)
        - x is of shape (in_features x 1)
        - compute gradient of loss wrt parameters and perform parameter update
        - compute gradient of loss wrt input and return
        """

        # compute gradient of loss wrt weights matrix given grad loss wrt layer output and input values
        grad_w = torch.matmul(grad_wrt_out, x.T)

        # cache gradient
        self.grad_w = grad_w

        # propagate gradient of loss wrt bias and cache
        if self.has_bias:
            grad_b = grad_wrt_out
            self.grad_b = grad_b
        
        # compute gradient of loss wrt layer inputs
        grad_x = torch.matmul(self.weight.T, grad_wrt_out)

        # update weights matrix
        #self.weight -= 1e-5 * grad_w

        # update bias
        #if self.has_bias:
            #self.bias -= 1e-5 * grad_b

        # return grad loss wrt layer inputs
        return grad_x
    
class ReLU():

    def __init__(self):
        self.last_in = None

    def forward(self, x):
        """
        Forward to compute output of layer given layer input
        - x is shape (in_features x 1)
        """

        # zero out all non-zero entries
        self.last_in = x
        f = x.clone()
        f[f < 0] = 0

        return f
    
    def backward(self, grad_wrt_out, x):
        """
        Backward to compute grad_wrt_input (in_features x 1) from grad_wrt_output (out_features x 1)
        """
        return grad_wrt_out * (x > 0)


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

class LayeredModel():

    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)

    def forward(self, x):
        activation = x
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation
    
    def backward(self, grad_loss_wrt_model_out):
        last_gradient = grad_loss_wrt_model_out
        for i in range(self.num_layers):
            layer = self.layers[self.num_layers - 1 - i]
            last_gradient = layer.backward(
                last_gradient, layer.last_in
            )