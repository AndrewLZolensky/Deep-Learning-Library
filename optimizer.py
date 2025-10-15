from layers import LinearLayer, LayeredModel
import torch

class Optimizer():
    def __init__(self, lr, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay
    def optimize(self, model):
        """
        Update model using output and loss fn

        - Assumes model.backward() has already been called --> model.grad_<param> attributes have been filled
        """

        # logic for LayeredModel
        if isinstance(model, LayeredModel):

            # perform gradient update for all model sub-layers (recurse)
            for layer in model.layers:
                self.optimize(layer)
        
        # logic for individual layer
        if isinstance(model, LinearLayer):

            # perform gradient update for linear layer
            model.weight -= (self.lr * model.grad_w + self.weight_decay * model.weight)
            if model.has_bias:
                model.bias -= self.lr * model.grad_b
        