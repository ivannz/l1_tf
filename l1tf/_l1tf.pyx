# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

# Author: Ivan Nazarov <ivannnnz@gmail.com>

import numpy as np
cimport numpy as np
cimport cython

from cpython cimport bool

np.import_array()

cdef extern from "l1_tf/l1tf.c":
    # had to change `lambda` to `C`
    int c_l1tf "l1tf" (const int n, const double *y, const double C, double *x,
                       const int verbose) nogil
    double c_l1tf_Cmax "l1tf_lambdamax"(const int n, double *y,
                                        const int verbose) nogil

def l1tf(double[::1] X, double C, bool rel_c):
    """Driver function for L1 trend filtering via primal-dual algorithm by
    Kwangmoo Koh, Seung-Jean Kim, and Stephen Boyd (C version 0.7 Aug 18 2007,
    cf. http://stanford.edu/~boyd/l1_tf/).

    Parameters
    ----------
    X: pd.Series or pd.DataFrame
        A 1D or flat data container wit hbuffer interface.
    C: float
        The L1 regularization coefficient (`lambda`).
    rel_c: boolean
        Determines whether the specified `C` an adaptively chosen maximum.

    Returns
    -------
    trend : arraylike
        The trend inferred by the `l1 trend filtering via primal-dual algorithm`.
        If `ary` is a Pandas object then its names/columns and index are preserved.
    """

    cdef double C_max
    cdef np.intp_t n_samples = X.shape[0]
    cdef np.double_t[::1] output_ = np.empty(n_samples, dtype=np.double)

    if rel_c:
        C_max = c_l1tf_Cmax(n_samples, &X[0], 0)
        if C_max < 0:
            raise RuntimeError("""Cannot solve for adaptive regularization parameter.""")
        C *= C_max

    with nogil:
        c_l1tf(n_samples, &X[0], C, &output_[0], 0)

    return output_

def l1tf_Cmax(double[::1] X):
    """A helper rotine for adaptive `C` value selection.
    Parameters
    ----------
    X: pd.Series or pd.DataFrame
        A 1D or flat data container wit hbuffer interface.

    Returns
    -------
    C_max : float
        The largest adaptive `C` value.
    """

    cdef double C_max = c_l1tf_Cmax(X.shape[0], &X[0], 0)
    if C_max < 0:
        raise RuntimeError("""Cannot solve for adaptive regularization parameter.""")
    return C_max
