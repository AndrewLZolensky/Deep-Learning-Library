import torch
import matplotlib.pyplot as plt
from layers import LinearLayer, ReLU, MSELossLayer, LayeredModel
from optimizer import Optimizer
import math

def generate_data(num_samples, num_in_features, num_out_features):

    # sample N inputs from standard multivariate gaussian
    samples = torch.randn((num_in_features, num_samples))

    # create sample neural network
    l1 = LinearLayer(num_in_features, 32)
    l2 = LinearLayer(32, num_out_features)
    r = ReLU()

    # do inference
    x1 = r.forward(l1.forward(samples))
    x2 = r.forward(l2.forward(x1))

    # return data
    return samples, x2

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

    

# generate some toy data
x, y = generate_data(1000, 16, 4)
print(x.shape, y.shape)

# split into train, val, test sets
xtrain, ytrain, xval, yval, xtest, ytest = split(x, y, 0.7, 0.15, 0.15)

# create model
loss = MSELossLayer()
model = LayeredModel([LinearLayer(16, 32), ReLU(), LinearLayer(32, 4), ReLU()])

# create optimizer
opt = Optimizer(lr = 1e-3, weight_decay=0.01)

# hold losses
losses = []

# set training hyper-params
shuffling = True
batch_size = 64

# get input data
for e in range(1000):

    # shuffle if desired
    if shuffling:
        perm = torch.randperm(x.shape[1])
        x_shuffled = x[:, perm]
        y_shuffled = y[:, perm]
    else:
        x_shuffled = x
        y_shuffled = y
    
    for i in range(math.ceil(x.shape[1] / batch_size)):

        # get single x, y example
        x_in = x_shuffled[:, i*batch_size:i*batch_size+batch_size]
        y_target = y_shuffled[:, i*batch_size:i*batch_size+batch_size]

        # compute model prediction
        y_pred = model.forward(x_in)

        # compute loss value and store
        obj = loss.forward(y_pred, y_target)

        # compute gradient of loss wrt model output
        grad_loss_wrt_model_output = loss.backward(y_pred, y_target)

        # compute gradient of loss wrt model parameters
        model.backward(grad_loss_wrt_model_output)

        # optimize model using gradient of loss wrt model output
        opt.optimize(model)
    
    # print loss value
    print(obj)
    losses.append(obj)

# plot training loss history
plt.plot(losses)
plt.show()