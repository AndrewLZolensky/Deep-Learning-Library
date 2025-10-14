import torch
import matplotlib.pyplot as plt
from layers import LinearLayer, ReLU, MSELossLayer

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
#print(xtrain.shape, ytrain.shape, xval.shape, yval.shape, xtest.shape, ytest.shape)

# set shuffling to T/F
shuffling = False

# create model
l1 = LinearLayer(16, 32)
l2 = LinearLayer(32, 4)
r = ReLU()
loss = MSELossLayer()

# hold losses
losses = []

# set learning rate
lr = 1e-5

# get input data
for e in range(100):

    # shuffle if desired
    if shuffling:
        perm = torch.randperm(x.shape[1])
        x_shuffled = x[:, perm]
        y_shuffled = y[:, perm]
    else:
        x_shuffled = x
        y_shuffled = y
    
    for i in range(x.shape[1]):
        x_in = x_shuffled[:, i].unsqueeze(-1)
        y_in = y_shuffled[:,i].unsqueeze(-1)
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
        grad_loss_z1 = l2.backward(grad_loss_h2, z1)

        l2.weight -= 1e-5 * l2.grad_w
        if l2.has_bias:
            l2.bias -= -1e-5 * l2.grad_b

        # compute grad loss wrt h1
        grad_loss_h1 = r.backward(grad_loss_z1, h1)

        # compute grad loss wrt x_in and update linear layer
        grad_loss_x_in = l1.backward(grad_loss_h1, x_in)

        # implement weight update
        l1.weight -= lr * l1.grad_w
        if l1.has_bias:
            l1.bias -= lr * l1.grad_b

    print(obj)
    losses.append(obj)

plt.plot(losses)
plt.show()