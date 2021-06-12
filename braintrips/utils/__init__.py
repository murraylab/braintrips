""" Auxiliary functions. """

import numpy as np
from scipy.special import erf
from scipy.stats import mstats
from scipy.stats import t, norm
from math import atanh, pow
from numpy import tanh


def loading(a, b):
    """
    Model-empirical loading.

    Parameters
    ----------
    a : array_like
        vector 1
    b : array_like
        vector 2

    Returns
    -------
    x : float
        normalized dot product

    """
    return a.dot(b)/np.power(norm(b), 2), None


# ---------------
# Transformations
# ---------------

def zscore(x):
    x -= x.mean()
    x /= x.std()
    return x


def zscore_df(x):
    """ Z-score rows of a dataframe.

    Parameters
    ----------
    x : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        z-scored dataframe

    """
    return x.subtract(x.mean(axis=1), axis=0).divide(x.std(axis=1), axis=0)


# -----------------
# Statistical tests
# -----------------

def pearsonr_multi(x, y):
    """
    Multi-dimensional Pearson correlation coefficient.

    Parameters
    ----------
    x : (N,P) np.ndarray
    y : (M,P) np.ndarray

    Returns
    -------
    (N,M) np.ndarray

    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must be same size in 2nd dimension.')

    mu_x = x.mean(axis=1)
    mu_y = y.mean(axis=1)

    s_x = x.std(axis=1, ddof=n - 1)
    s_y = y.std(axis=1, ddof=n - 1)
    cov = np.dot(x, y.T) - n * np.dot(
        mu_x[:, np.newaxis], mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def theil_sen(x, y):
    """ Nonparametric Theil-Sen estimator of linear slope. """
    xz = zscore(x)
    yz = zscore(y)
    mt, bt, lower, upper = mstats.theilslopes(yz, xz)

    assert np.allclose(bt, np.median(yz) - mt * np.median(xz))

    sigma_ratio = y.std() / x.std()
    mprime = mt * sigma_ratio
    bprime = y.mean() + (y.std() * bt) - (mprime * x.mean())

    mprime_l = lower * sigma_ratio
    b_theil_l = np.median(yz) - lower * np.median(xz)
    bprime_l = y.mean() + (y.std() * b_theil_l) - (
            mprime_l * x.mean())

    mprime_u = upper * sigma_ratio
    b_theil_u = np.median(yz) - upper * np.median(xz)
    bprime_u = y.mean() + (y.std() * b_theil_u) - (
            mprime_u * x.mean())

    return (mprime, mprime_l, mprime_u), (bprime, bprime_l, bprime_u)


def rz_ci(r, n, conf_level=0.95):
    zr_se = pow(1/(n - 3), .5)
    moe = norm.ppf(1 - (1 - conf_level)/float(2)) * zr_se
    zu = atanh(r) + moe
    zl = atanh(r) - moe
    return tanh((zl, zu))


def rho_rxy_rxz(rxy, rxz, ryz):
    num = (ryz-1/2.*rxy*rxz)*(1-pow(rxy, 2)-pow(rxz, 2)-pow(ryz, 2))+pow(ryz, 3)
    den = (1 - pow(rxy, 2)) * (1 - pow(rxz, 2))
    return num/float(den)


def dependent_corr(xy, xz, yz, n, twotailed=True, conf_level=0.95,
                   method='steiger'):
    """
    Calculates significance between two dependent correlation coefficients
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between x and z
    @param yz: correlation coefficient between y and z
    @param n: number of elements in x, y and z
    @param twotailed: whether to calculate a one or two tailed test, only works
     for 'steiger' method
    @param conf_level: confidence level, only works for 'zou' method
    @param method: defines the method uses, 'steiger' or 'zou'
    @return: t and p-val
    """
    if method == 'steiger':
        d = xy - xz
        determin = 1 - xy * xy - xz * xz - yz * yz + 2 * xy * xz * yz
        av = (xy + xz)/2
        cube = (1 - yz) * (1 - yz) * (1 - yz)

        t2 = d * np.sqrt((n - 1) * (1 + yz) / (
                    (2 * (n - 1) / (n - 3)) * determin + av * av * cube))
        p = 1 - t.cdf(abs(t2), n - 3)

        if twotailed:
            p *= 2

        return t2, p
    elif method == 'zou':
        L1 = rz_ci(xy, n, conf_level=conf_level)[0]
        U1 = rz_ci(xy, n, conf_level=conf_level)[1]
        L2 = rz_ci(xz, n, conf_level=conf_level)[0]
        U2 = rz_ci(xz, n, conf_level=conf_level)[1]
        rho_r12_r13 = rho_rxy_rxz(xy, xz, yz)
        lower = xy - xz - pow((pow((xy - L1), 2) + pow(
            (U2 - xz), 2) - 2 * rho_r12_r13 * (xy - L1) * (U2 - xz)), 0.5)
        upper = xy - xz + pow((pow((U1 - xy), 2) + pow(
            (xz - L2), 2) - 2 * rho_r12_r13 * (U1 - xy) * (xz - L2)), 0.5)
        return lower, upper
    else:
        raise Exception('Wrong method!')


def nonparametric_pvalue(test_stat, null_dist, two_tailed=True):
    """
    Compute p-value nonparametrically using null distribution.

    Parameters
    ----------
    test_stat : scalar
        test statistic
    null_dist : (, N) np.ndarray
        null distribution
    two_tailed : bool (optional, default True)
        compute two-tailed significance value

    Returns
    -------
    float
        number of (one- or two-sided) samples in the null distribution more
        extreme than the test statistic
    """
    if two_tailed:
        n_more_extreme = np.greater(np.abs(null_dist), np.abs(test_stat)).sum()
    else:
        if test_stat > 0:
            n_more_extreme = np.greater(null_dist, test_stat).sum()
        else:
            n_more_extreme = np.less(null_dist, test_stat).sum()
    return n_more_extreme / float(null_dist.size)


def linearize_map(x):
    """
    linearization via error function transformation
    """
    return erf((x - np.mean(x)) / x.std() / np.sqrt(2))

# -------------------
# Surrogate map utils
# -------------------


def compare_fc(model, empirical):
    """
    Compute similarity between upper-triangular elements of two FC matrices.

    Parameters
    ----------
    model : (N, N) np.ndarray
        model FC
    empirical : (N, N) np.ndarray
        empirical FC

    Returns
    -------
    float
    """
    triu_inds = np.triu_indices(180, k=1)
    return metric(model[triu_inds], empirical[triu_inds])
