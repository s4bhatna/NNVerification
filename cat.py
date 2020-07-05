import numpy as np

def transform_input(x):
    """
    x: an input matrix with dimensions N x M

    Outputs a vector x_v of dimension N*M x 1

    """
    
    nrow = x.shape[0]
    ncol = x.shape[1]

    dim = nrow*ncol

    x_v = x.reshape(dim)

    return x_v


def relu(ivect):
    """
    ivect: an input vector similar to output of transform_input

    Applies the ReLU() activation function as a composition
    as described in CAT representation

    """

    dim  = ivect.shape[0]
    out = ivect

    for i in range(dim):

        M = np.eye(dim)

        if out[i] <= 0:
            M[:,i] = np.zeros(dim)
            out = np.matmul(M, out)
        else:
            out = out

    return out

def fully_connected(W, b, X):
    """
    W: a vector of weights 
    b: bias vector
    X: transformed input vector derived form input matrix

    Returns the output of a fully connected layer and
    applies the ReLU activation
    """
    affine = np.matmul(W, X) + b
    fc = relu(affine)
    return fc

def conv(W, x_mat):
    
    nrow = x_mat.shape[0]
    ncol = x_mat.shape[1]
    x = x_mat.reshape(nrow*ncol)
    
    conv_out = []
    
    for j in range(ncol-1):
        for i in range(nrow-1):
            x_conv = np.array([x[nrow*j+i], x[nrow*j+i+1], x[nrow*j+i+nrow], x[nrow*j+i+nrow+1]])
            conv_out.append(np.matmul(W, x_conv))
    
    return conv_out

def conv_weights(W, x_mat):

    """
    W: a vector of weights
    x_mat: input in matrix form

    Out puts the weight matrix W_F to simulate convolution.
    Currently only works for 2 x 2 stride.
    """
    
    nrow = x_mat.shape[0]
    ncol = x_mat.shape[1]
    x = x_mat.reshape(nrow*ncol)
    
    W_len = len(W)
    dim_1 = nrow*ncol
    dim_2 = len(conv(W, x_mat))
    weights = []
    
    w_idxs = []
    
    for j in range(ncol-1):
        for i in range(nrow-1):
            x_conv = np.array([nrow*j+i, nrow*j+i+1, nrow*j+i+nrow, nrow*j+i+nrow+1])
            w_idxs.append(x_conv)
        
    for idx_array in w_idxs:
        
        w_row = np.zeros(dim_1)
        
        for idx in range(W_len):
            w_idx = idx_array[idx]
            w_row[w_idx] = W[idx]
        
        weights.append(w_row)
    
    weights = np.stack(weights, axis=0)
        
    return weights