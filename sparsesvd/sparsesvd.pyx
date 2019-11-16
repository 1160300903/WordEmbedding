# cython: infer_types=True

from scipy.sparse import issparse, isspmatrix_csc
cimport numpy as np
import numpy as np
from libc.stdlib cimport free

def sparsesvd(X, D, k):
    if not isspmatrix_csc(X):
        raise TypeError("First argument must be a scipy.sparse.csc_matrix")
    if not isspmatrix_csc(D):
        raise TypeError("Second argument must be a scipy.sparse.csc_matrix")
    k = int(k)

    cdef SVDRec srec
    cdef smat matX
    cdef smat matD

    matX.rows = X.shape[0]
    matX.cols = X.shape[1]
    matX.vals = X.nnz

    matD.rows = D.shape[0]
    matD.cols = D.shape[1]
    matD.vals = D.nnz

    cdef long [:] indptrX = np.ascontiguousarray(X.indptr, dtype=np.dtype('l'))
    cdef long [:] indicesX = np.ascontiguousarray(X.indices, dtype=np.dtype('l'))
    cdef double [:] dataX = np.ascontiguousarray(X.data, dtype=np.double)
    matX.pointr = &indptrX[0]
    matX.rowind = &indicesX[0]
    matX.value = &dataX[0]

    cdef long [:] indptrD = np.ascontiguousarray(D.indptr, dtype=np.dtype('l'))
    cdef long [:] indicesD = np.ascontiguousarray(D.indices, dtype=np.dtype('l'))
    cdef double [:] dataD = np.ascontiguousarray(D.data, dtype=np.double)
    matD.pointr = &indptrD[0]
    matD.rowind = &indicesD[0]
    matD.value = &dataD[0]

    srec = svdLAS2A(&matX, &matD, k)

    p_Ut = <double[:srec.d, :srec.Ut.cols]> srec.Ut.value[0]
    p_Ut.callback_free_data = free
    Ut = np.array(p_Ut, copy=False)

    p_Vt = <double[:srec.d, :srec.Vt.cols]> srec.Vt.value[0]
    p_Vt.callback_free_data = free
    Vt = np.array(p_Vt, copy=False)

    p_s = <double[:srec.d]> srec.S
    p_s.callback_free_data = free
    s = np.array(p_s, copy=False)

    # this was malloc'ed by svdLAS2A
    free(srec.Ut.value)
    free(srec.Ut)
    free(srec.Vt)
    free(srec.Vt.value)
    free(srec)

    return Ut, s, Vt
