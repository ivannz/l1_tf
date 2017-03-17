"""A wrapper for L1 trend filtering via primal-dual algorithm by
Kwangmoo Koh, Seung-Jean Kim, and Stephen Boyd (C version 0.7 Aug 18 2007,
cf. http://stanford.edu/~boyd/l1_tf/).
"""
import numpy as np
import pandas as pd

from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin

from ._l1tf import l1tf as c_l1tf, l1tf_Cmax as c_l1tf_Cmax

class L1TrendFilter(BaseEstimator, TransformerMixin):
    """L1 trend filtering via primal-dual algorithm by Kwangmoo Koh,
    Seung-Jean Kim, Stephen Boyd, and Dimitry Gorinevsky.

    References
    ----------
    .. [1] Kim SJ, Koh K, Boyd S, Gorinevsky D. "l_1 Trend Filtering".
           SIAM review. 2009 May 4;51(2):339-60. doi:10.1137/070690274
           http://stanford.edu/~boyd/l1_tf/
    """
    def __init__(self, C=None, relative=False):
        """L1 trend filtering via primal-dual algorithm.

        Parameters
        ----------
        C: None, or float
            The L1 regularization coefficient (`lambda`).
        relative: boolean, default=False
            Determines whether the specifiec `C` is relative to the EIGENVALUE.
        l1tf_path: str, or None
            Path to folder containing `l1tf` excutable.
        tmp_dir: str, default=None
            The directory to keep temporary files in. `Nonez means the system `/tmp`
            folder or analogue.
        """
        super(L1TrendFilter, self).__init__()
        self.relative = relative
        self.C = C

    def fit(self, X, y=None):
        """Computes, if necessary, the maximal adaptive values of
        the regularization parameter `C`.
        """
        if self.C is not None and not isinstance(self.C, (float, int)):
            raise TypeError("""`C` should be either `None` or a float.""")

        C_ = 0.1 if self.C is None else float(self.C)
        if C_ < 0:
            raise ValueError("""`C` must be nonnegative.""")

        # Ensure `X` is a 2d numeric array: it is important that
        #  the 2d array be in column-major layout.
        ary2d = check_array(X, accept_sparse=False, dtype="numeric", order="F",
                            allow_nd=False, ensure_min_samples=1)
        self.C_max_ = [c_l1tf_Cmax(ary2d[:, j])
                       for j in xrange(ary2d.shape[1])]
        self.C_ = C_
        return self
    
    def transform(self, X):
        """Extracts the L1 trend (continuous piecewise linear trend) from
        the input data.
        """
        # Ensure `X` is a 2d numeric array: it is important that
        #  the 2d array be in column-major layout.
        ary2d = check_array(X, accept_sparse=False, dtype="numeric", order="F",
                            allow_nd=False, ensure_min_samples=1)
        if len(self.C_max_) != X.shape[1]:
            raise TypeError("""The input data must have the same dimension """
                            """as the train data.""")

        # Perform the API call
        trend = np.empty_like(ary2d, dtype=np.float)
        for j, C_max_ in enumerate(self.C_max_):
            C_ = self.C_ * C_max_ if self.relative else self.C_  
            trend[:, j] = c_l1tf(ary2d[:, j], C=C_, rel_c=False)

        # Assemble a Pandas object if one was givenz
        if isinstance(X, pd.DataFrame):
            trend = type(X)(trend, index=X.index, columns=X.columns)
        elif isinstance(X, pd.Series):
            trend = pd.Series(trend, index=X.index, name=X.name)
        return trend

def l1_filter(ary, C=None, relative=False):
    """Driver function for L1 trend filtering via primal-dual algorithm by
    Kwangmoo Koh, Seung-Jean Kim, and Stephen Boyd (C version 0.7 Aug 18 2007,
    cf. http://stanford.edu/~boyd/l1_tf/).

    Parameters
    ----------
    ary: pd.Series or pd.DataFrame
        The 1d (or flat) data for filtering.
    C: None, or float
        The L1 regularization coefficient (`lambda`).
    relative: boolean, default=False
        Determines whether the specifiec `C` is relative to the EIGENVALUE.

    Returns
    -------
    trend : arraylike
        The trend inferred by the `l1 trend filtering via primal-dual algorithm`.
        If `ary` is a Pandas object then its names/columns and index are preserved.
    """
    # check input
    ary_ = np.asanyarray(ary)
    if ary_.ndim > 1:
        raise TypeError("""`ary` must be either a 1d array-like or flat.""")

    if C is None:
        C = 0.1
    elif not (isinstance(C, (float, int)) and C >= 0):
        raise ValueError("""`C` must be a nonegative scalar.""")

    # perform the API call
    trend = c_l1tf(ary_, C=C, rel_c=relative).base

    if isinstance(ary, pd.DataFrame):
        trend = type(ary)(trend, index=ary.index, columns=ary.columns)
    elif isinstance(ary, pd.Series):
        trend = pd.Series(trend, index=ary.index, name=ary.name)
    return trend
