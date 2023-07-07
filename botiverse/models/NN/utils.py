import numpy as np

def split_data(X,y, val_ratio):
    '''
    Split the data into training and validation sets.
    '''
    N = X.shape[0]
    # randomly shuffle the data with numpy's permutation function.
    index = np.random.permutation(N)
    X, y = X[index], y[index]
    # split
    V = int(N * val_ratio)
    X_t, y_t = X[:N - V], y[:N - V]
    X_v, y_v = X[N - V:], y[N - V:]
    return X_t, y_t, X_v, y_v


def batchify(x_data, y_data, batch_size):
    '''
    Given x_data of shape (N, d) and y_data of shape (N) return x_data of shape (B, N//B, d, 1) and y_data of shape (B, N//B, K, 1).
    '''
    N = x_data.shape[0]
    # shuffle the data with numpy's permutation function.
    index = np.random.permutation(N)
    x_data, y_data = x_data[index], y_data[index]
    
    # add a trailing dimension to x_data
    x_data = x_data[..., np.newaxis]
    
    # make y_data a one-hot vector
    u = len(np.unique(y_data))
    y_data = np.array([np.identity(u)[:,[y]] for y in y_data])           #y is a one-hot column vector.
    
    # truncate the data to be divisible by the batch size
    x_data = x_data[:N - N%batch_size]
    y_data = y_data[:N - N%batch_size]
    
    # batchify the data using numpy
    x_data = np.split(x_data, N//batch_size)
    y_data = np.split(y_data, N//batch_size)
    
    return np.array(x_data), np.array(y_data)