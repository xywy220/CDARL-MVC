import logging
import numpy as np
import math
import torch

def get_logger():
    """Get logging."""
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

class Gaussian_sampler(object):
    def __init__(self, mean, sd=1, N=20000):
        self.total_size = N
        self.mean = mean
        self.sd = sd
        np.random.seed(1024)
        self.X = np.random.normal(self.mean, self.sd, (self.total_size,len(self.mean))) #[N, dim]
        self.X = self.X.astype('float32')
        self.Y = None

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X[indx, :]

    def get_batch(self,batch_size):
        return np.random.normal(self.mean, self.sd, (batch_size,len(self.mean))).astype('float32')

    def load_all(self):
        return self.X, self.Y

def next_batch(X1, X2, X3, X4, mask, batch_size):
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]
        batch_x4 = X4[start_idx: end_idx, ...]
        batch_x5 = mask[start_idx: end_idx, ...]

        yield (batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, (i + 1))

def next_batch_multiviews(X, mask, batch_size, views):
    tot = X['X1'][0].shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        batch_x, batch_idx = [], []
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        for j in range(views):
            batch_x.append(X['X1'][j][start_idx: end_idx, ...])
            batch_idx.append(X[f'X{j+2}'][start_idx: end_idx, ...])
        batch_m = mask[start_idx: end_idx, ...]

        yield (batch_x, batch_idx, batch_m, (i + 1))

def cal_std(logger, *arg):
    """Return the average and its std"""
    if len(arg) == 3:
        print('ACC:'+ str(arg[0]))
        print('NMI:'+ str(arg[1]))
        print('ARI:'+ str(arg[2]))
        output = "ACC {:.3f} std {:.3f} NMI {:.3f} std {:.3f} ARI {:.3f} std {:.3f}".format( np.mean(arg[0]),
                                                                                             np.std(arg[0]),
                                                                                             np.mean(arg[1]),
                                                                                             np.std(arg[1]),
                                                                                             np.mean(arg[2]),
                                                                                             np.std(arg[2]))
    elif len(arg) == 1:
        print(arg)
        output = "ACC {:.3f} std {:.3f}".format(np.mean(arg), np.std(arg))

    print(output)
    return

def normalize(x):
    """Normalize"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist
