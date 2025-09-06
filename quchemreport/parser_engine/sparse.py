import numpy as np
import sklearn
import scipy.sparse as sp

def sparse(data, mo_coefs, nb_coef, b_cut):
    threshold = 0.05 # compression with loss threshold
    for a in data.mocoeffs[nb_coef:]:
        # normalization
        a_ = sklearn.preprocessing.normalize(np.abs(a), norm='l1', copy=False)
        # indices of sorting and sorting
        a_argsort = a_.argsort(1)
        a_.sort(axis=1)
        az = np.where(a_.cumsum(axis=1) < threshold )
        # zeroing
        a[az[0], a_argsort[az]] = 0.
        a = a[:b_cut, :]
        # to sparse csr matrix
        acsr = sp.csr_matrix(a)
        # append tuple for the csr to mo_coefs
        mo_coefs.append( (acsr.data.tolist(), acsr.indices.tolist(), acsr.indptr.tolist()) )