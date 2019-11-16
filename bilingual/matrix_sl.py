import numpy as np
from scipy.sparse import csc_matrix
def save_matrix(f, m):
    np.savez_compressed(f, data=m.data, indices=m.indices, indptr=m.indptr, shape=m.shape)


def load_matrix(f):
    if not f.endswith('.npz'):
        f += '.npz'
    loader = np.load(f)
    return csc_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])