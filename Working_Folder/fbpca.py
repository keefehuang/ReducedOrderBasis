"""
Functions for principal component analysis (PCA) and accuracy checks

---------------------------------------------------------------------

This module contains eight functions:

pca
    principal component analysis (singular value decomposition)
eigens
    eigendecomposition of a self-adjoint matrix
eigenn
    eigendecomposition of a nonnegative-definite self-adjoint matrix
diffsnorm
    spectral-norm accuracy of a singular value decomposition
diffsnormc
    spectral-norm accuracy of a centered singular value decomposition
diffsnorms
    spectral-norm accuracy of a Schur decomposition
mult
    default matrix multiplication
set_matrix_mult
    re-definition of the matrix multiplication function "mult"

---------------------------------------------------------------------

Copyright 2014 Facebook Inc.
All rights reserved.

"Software" means the fbpca software distributed by Facebook Inc.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in
  the documentation and/or other materials provided with the
  distribution.

* Neither the name Facebook nor the names of its contributors may be
  used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Additional grant of patent rights:

Facebook hereby grants you a perpetual, worldwide, royalty-free,
non-exclusive, irrevocable (subject to the termination provision
below) license under any rights in any patent claims owned by
Facebook, to make, have made, use, sell, offer to sell, import, and
otherwise transfer the Software. For avoidance of doubt, no license
is granted under Facebook's rights in any patent claims that are
infringed by (i) modifications to the Software made by you or a third
party, or (ii) the Software in combination with any software or other
technology provided by you or a third party.

The license granted hereunder will terminate, automatically and
without notice, for anyone that makes any claim (including by filing
any lawsuit, assertion, or other action) alleging (a) direct,
indirect, or contributory infringement or inducement to infringe any
patent: (i) by Facebook or any of its subsidiaries or affiliates,
whether or not such claim is related to the Software, (ii) by any
party if such claim arises in whole or in part from any software,
product or service of Facebook or any of its subsidiaries or
affiliates, whether or not such claim is related to the Software, or
(iii) by any party relating to the Software; or (b) that any right in
any patent claim of Facebook is invalid or unenforceable.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import unittest
import math
import numpy as np
from numpy.fft import rfft
from numpy.linalg import qr, pinv
from scipy.linalg import cholesky, eigh, lu, svd, norm, solve, hadamard, qr as spqr
from scipy.sparse import coo_matrix, issparse, spdiags, block_diag
from copy import deepcopy
import gc
from math import ceil
#import time


def diffsnorm(A, U, s, Va, n_iter=20):
    """
    2-norm accuracy of an approx to a matrix.

    Computes an estimated norm of the spectral norm (the operator norm
    induced by the Euclidean vector norm) of A - U diag(s) Va, using
    n_iter iterations of the power method started with a random vector;
    n_iter must be a positive integer.

    Increasing n_iter improves the accuracy of the estimate snorm of
    the spectral norm of A - U diag(s) Va.

    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    Parameters
    ----------
    A : array_like
        first matrix in A - U diag(s) Va whose spectral norm is being
        estimated
    U : array_like
        second matrix in A - U diag(s) Va whose spectral norm is being
        estimated
    s : array_like
        vector in A - U diag(s) Va whose spectral norm is being
        estimated
    Va : array_like
        fourth matrix in A - U diag(s) Va whose spectral norm is being
        estimated
    n_iter : int, optional
        number of iterations of the power method to conduct;
        n_iter must be a positive integer, and defaults to 20

    Returns
    -------
    float
        an estimate of the spectral norm of A - U diag(s) Va (the
        estimate fails to be accurate with exponentially low prob. as
        n_iter increases; see references DM1_, DM2_, and DM3_ below)

    Examples
    --------
    >>> from fbpca import diffsnorm, pca
    >>> from numpy.random import uniform
    >>> from scipy.linalg import svd
    >>>
    >>> A = uniform(low=-1.0, high=1.0, size=(100, 2))
    >>> A = A.dot(uniform(low=-1.0, high=1.0, size=(2, 100)))
    >>> (U, s, Va) = svd(A, full_matrices=False)
    >>> A = A / s[0]
    >>>
    >>> (U, s, Va) = pca(A, 2, True)
    >>> err = diffsnorm(A, U, s, Va)
    >>> print(err)

    This example produces a rank-2 approximation U diag(s) Va to A such
    that the columns of U are orthonormal, as are the rows of Va, and
    the entries of s are all nonnegative and are nonincreasing.
    diffsnorm(A, U, s, Va) outputs an estimate of the spectral norm of
    A - U diag(s) Va, which should be close to the machine precision.

    References
    ----------
    .. [DM1] Jacek Kuczynski and Henryk Wozniakowski, Estimating the
             largest eigenvalues by the power and Lanczos methods with
             a random start, SIAM Journal on Matrix Analysis and
             Applications, 13 (4): 1094-1122, 1992.
    .. [DM2] Edo Liberty, Franco Woolfe, Per-Gunnar Martinsson,
             Vladimir Rokhlin, and Mark Tygert, Randomized algorithms
             for the low-rank approximation of matrices, Proceedings of
             the National Academy of Sciences (USA), 104 (51):
             20167-20172, 2007. (See the appendix.)
    .. [DM3] Franco Woolfe, Edo Liberty, Vladimir Rokhlin, and Mark
             Tygert, A fast randomized algorithm for the approximation
             of matrices, Applied and Computational Harmonic Analysis,
             25 (3): 335-366, 2008. (See Section 3.4.)

    See also
    --------
    diffsnormc, pca
    """

    m, n = np.shape(A)
    (m2, k) = U.shape
    k2 = len(s)
    l = len(s)
    (l2, n2) = Va.shape

    assert m == m2
    assert k == k2
    assert l == l2
    assert n == n2

    assert n_iter >= 1

    if np.isrealobj(A) and np.isrealobj(U) and np.isrealobj(s) and \
            np.isrealobj(Va):
        isreal = True
    else:
        isreal = False

    if m >= n:

        #
        # Generate a random vector x.
        #
        if isreal:
            x = np.random.normal(size=(n, 1))
        else:
            x = np.random.normal(size=(n, 1)) \
                + 1j * np.random.normal(size=(n, 1))

        x = x / norm(x)

        #
        # Run n_iter iterations of the power method.
        #
        for it in range(n_iter):
            #
            # Set y = (A - U diag(s) Va)x.
            #
            y = mult(A, x) - U.dot(np.diag(s).dot(Va.dot(x)))
            #
            # Set x = (A' - Va' diag(s)' U')y.
            #
            x = mult(y.conj().T, A).conj().T \
                - Va.conj().T.dot(np.diag(s).conj().T.dot(U.conj().T.dot(y)))

            #
            # Normalize x, memorizing its Euclidean norm.
            #
            snorm = norm(x)
            if snorm == 0:
                return 0
            x = x / snorm

        snorm = math.sqrt(snorm)

    if m < n:

        #
        # Generate a random vector y.
        #
        if isreal:
            y = np.random.normal(size=(m, 1))
        else:
            y = np.random.normal(size=(m, 1)) \
                + 1j * np.random.normal(size=(m, 1))

        y = y / norm(y)

        #
        # Run n_iter iterations of the power method.
        #
        for it in range(n_iter):
            #
            # Set x = (A' - Va' diag(s)' U')y.
            #
            x = mult(y.conj().T, A).conj().T \
                - Va.conj().T.dot(np.diag(s).conj().T.dot(U.conj().T.dot(y)))
            #
            # Set y = (A - U diag(s) Va)x.
            #
            y = mult(A, x) - U.dot(np.diag(s).dot(Va.dot(x)))

            #
            # Normalize y, memorizing its Euclidean norm.
            #
            snorm = norm(y)
            if snorm == 0:
                return 0
            y = y / snorm

        snorm = math.sqrt(snorm)

    return snorm


class TestDiffsnorm(unittest.TestCase):

    def test_dense(self):

        logging.info('running TestDiffsnorm.test_dense...')
        logging.info('err =')

        for (m, n) in [(200, 100), (100, 200), (100, 100)]:
            for isreal in [True, False]:

                if isreal:
                    A = np.random.normal(size=(m, n))
                if not isreal:
                    A = np.random.normal(size=(m, n)) \
                        + 1j * np.random.normal(size=(m, n))

                (U, s, Va) = svd(A, full_matrices=False)
                snorm = diffsnorm(A, U, s, Va)
                logging.info(snorm)
                self.assertTrue(snorm < .1e-10 * s[0])

    def test_sparse(self):

        logging.info('running TestDiffsnorm.test_sparse...')
        logging.info('err =')

        for (m, n) in [(200, 100), (100, 200), (100, 100)]:
            for isreal in [True, False]:

                if isreal:
                    A = 2 * spdiags(np.arange(min(m, n)) + 1, 0, m, n)
                if not isreal:
                    A = 2 * spdiags(np.arange(min(m, n)) + 1, 0, m, n) \
                        * (1 + 1j)

                A = A - spdiags(np.arange(min(m, n) + 1), 1, m, n)
                A = A - spdiags(np.arange(min(m, n)) + 1, -1, m, n)
                (U, s, Va) = svd(A.todense(), full_matrices=False)
                A = A / s[0]

                (U, s, Va) = svd(A.todense(), full_matrices=False)
                snorm = diffsnorm(A, U, s, Va)
                logging.info(snorm)
                self.assertTrue(snorm < .1e-10 * s[0])


def diffsnormc(A, U, s, Va, n_iter=20):
    """
    2-norm approx error to a matrix upon centering.

    Computes an estimate snorm of the spectral norm (the operator norm
    induced by the Euclidean vector norm) of C(A) - U diag(s) Va, using
    n_iter iterations of the power method started with a random vector,
    where C(A) refers to A from the input, after centering its columns;
    n_iter must be a positive integer.

    Increasing n_iter improves the accuracy of the estimate snorm of
    the spectral norm of C(A) - U diag(s) Va, where C(A) refers to A
    after centering its columns.

    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    Parameters
    ----------
    A : array_like
        first matrix in the column-centered A - U diag(s) Va whose
        spectral norm is being estimated
    U : array_like
        second matrix in the column-centered A - U diag(s) Va whose
        spectral norm is being estimated
    s : array_like
        vector in the column-centered A - U diag(s) Va whose spectral
        norm is being estimated
    Va : array_like
        fourth matrix in the column-centered A - U diag(s) Va whose
        spectral norm is being estimated
    n_iter : int, optional
        number of iterations of the power method to conduct;
        n_iter must be a positive integer, and defaults to 20

    Returns
    -------
    float
        an estimate of the spectral norm of the column-centered A
        - U diag(s) Va (the estimate fails to be accurate with
        exponentially low probability as n_iter increases; see
        references DC1_, DC2_, and DC3_ below)

    Examples
    --------
    >>> from fbpca import diffsnormc, pca
    >>> from numpy.random import uniform
    >>> from scipy.linalg import svd
    >>>
    >>> A = uniform(low=-1.0, high=1.0, size=(100, 2))
    >>> A = A.dot(uniform(low=-1.0, high=1.0, size=(2, 100)))
    >>> (U, s, Va) = svd(A, full_matrices=False)
    >>> A = A / s[0]
    >>>
    >>> (U, s, Va) = pca(A, 2, False)
    >>> err = diffsnormc(A, U, s, Va)
    >>> print(err)

    This example produces a rank-2 approximation U diag(s) Va to the
    column-centered A such that the columns of U are orthonormal, as
    are the rows of Va, and the entries of s are nonnegative and
    nonincreasing. diffsnormc(A, U, s, Va) outputs an estimate of the
    spectral norm of the column-centered A - U diag(s) Va, which
    should be close to the machine precision.

    References
    ----------
    .. [DC1] Jacek Kuczynski and Henryk Wozniakowski, Estimating the
             largest eigenvalues by the power and Lanczos methods with
             a random start, SIAM Journal on Matrix Analysis and
             Applications, 13 (4): 1094-1122, 1992.
    .. [DC2] Edo Liberty, Franco Woolfe, Per-Gunnar Martinsson,
             Vladimir Rokhlin, and Mark Tygert, Randomized algorithms
             for the low-rank approximation of matrices, Proceedings of
             the National Academy of Sciences (USA), 104 (51):
             20167-20172, 2007. (See the appendix.)
    .. [DC3] Franco Woolfe, Edo Liberty, Vladimir Rokhlin, and Mark
             Tygert, A fast randomized algorithm for the approximation
             of matrices, Applied and Computational Harmonic Analysis,
             25 (3): 335-366, 2008. (See Section 3.4.)

    See also
    --------
    diffsnorm, pca
    """

    m, n = np.shape(A)
    (m2, k) = U.shape
    k2 = len(s)
    l = len(s)
    (l2, n2) = Va.shape

    assert m == m2
    assert k == k2
    assert l == l2
    assert n == n2

    assert n_iter >= 1

    if np.isrealobj(A) and np.isrealobj(U) and np.isrealobj(s) and \
            np.isrealobj(Va):
        isreal = True
    else:
        isreal = False

    #
    # Calculate the average of the entries in every column.
    #
    c = A.sum(axis=0) / m
    c = c.reshape((1, n))

    if m >= n:

        #
        # Generate a random vector x.
        #
        if isreal:
            x = np.random.normal(size=(n, 1))
        else:
            x = np.random.normal(size=(n, 1)) \
                + 1j * np.random.normal(size=(n, 1))

        x = x / norm(x)

        #
        # Run n_iter iterations of the power method.
        #
        for it in range(n_iter):
            #
            # Set y = (A - ones(m,1)*c - U diag(s) Va)x.
            #
            y = mult(A, x) - np.ones((m, 1)).dot(c.dot(x)) \
                - U.dot(np.diag(s).dot(Va.dot(x)))
            #
            # Set x = (A' - c'*ones(1,m) - Va' diag(s)' U')y.
            #
            x = mult(y.conj().T, A).conj().T \
                - c.conj().T.dot(np.ones((1, m)).dot(y)) \
                - Va.conj().T.dot(np.diag(s).conj().T.dot(U.conj().T.dot(y)))

            #
            # Normalize x, memorizing its Euclidean norm.
            #
            snorm = norm(x)
            if snorm == 0:
                return 0
            x = x / snorm

        snorm = math.sqrt(snorm)

    if m < n:

        #
        # Generate a random vector y.
        #
        if isreal:
            y = np.random.normal(size=(m, 1))
        else:
            y = np.random.normal(size=(m, 1)) \
                + 1j * np.random.normal(size=(m, 1))

        y = y / norm(y)

        #
        # Run n_iter iterations of the power method.
        #
        for it in range(n_iter):
            #
            # Set x = (A' - c'*ones(1,m) - Va' diag(s)' U')y.
            #
            x = mult(y.conj().T, A).conj().T \
                - c.conj().T.dot(np.ones((1, m)).dot(y)) \
                - Va.conj().T.dot(np.diag(s).conj().T.dot(U.conj().T.dot(y)))
            #
            # Set y = (A - ones(m,1)*c - U diag(s) Va)x.
            #
            y = mult(A, x) - np.ones((m, 1)).dot(c.dot(x)) \
                - U.dot(np.diag(s).dot(Va.dot(x)))

            #
            # Normalize y, memorizing its Euclidean norm.
            #
            snorm = norm(y)
            if snorm == 0:
                return 0
            y = y / snorm

        snorm = math.sqrt(snorm)

    return snorm


class TestDiffsnormc(unittest.TestCase):

    def test_dense(self):

        logging.info('running TestDiffsnormc.test_dense...')
        logging.info('err =')

        for (m, n) in [(200, 100), (100, 200), (100, 100)]:
            for isreal in [True, False]:

                if isreal:
                    A = np.random.normal(size=(m, n))
                if not isreal:
                    A = np.random.normal(size=(m, n)) \
                        + 1j * np.random.normal(size=(m, n))

                c = A.sum(axis=0) / m
                c = c.reshape((1, n))
                Ac = A - np.ones((m, 1)).dot(c)

                (U, s, Va) = svd(Ac, full_matrices=False)
                snorm = diffsnormc(A, U, s, Va)
                logging.info(snorm)
                self.assertTrue(snorm < .1e-10 * s[0])

    def test_sparse(self):

        logging.info('running TestDiffsnormc.test_sparse...')
        logging.info('err =')

        for (m, n) in [(200, 100), (100, 200), (100, 100)]:
            for isreal in [True, False]:

                if isreal:
                    A = 2 * spdiags(np.arange(min(m, n)) + 1, 0, m, n)
                if not isreal:
                    A = 2 * spdiags(np.arange(min(m, n)) + 1, 0, m, n) \
                        * (1 + 1j)

                A = A - spdiags(np.arange(min(m, n) + 1), 1, m, n)
                A = A - spdiags(np.arange(min(m, n)) + 1, -1, m, n)
                (U, s, Va) = svd(A.todense(), full_matrices=False)
                A = A / s[0]

                Ac = A.todense()
                c = Ac.sum(axis=0) / m
                c = c.reshape((1, n))
                Ac = Ac - np.ones((m, 1)).dot(c)

                (U, s, Va) = svd(Ac, full_matrices=False)
                snorm = diffsnormc(A, U, s, Va)
                logging.info(snorm)
                self.assertTrue(snorm < .1e-10 * s[0])


def diffsnorms(A, S, V, n_iter=20):
    """
    2-norm accuracy of a Schur decomp. of a matrix.

    Computes an estimate snorm of the spectral norm (the operator norm
    induced by the Euclidean vector norm) of A-VSV', using n_iter
    iterations of the power method started with a random vector;
    n_iter must be a positive integer.

    Increasing n_iter improves the accuracy of the estimate snorm of
    the spectral norm of A-VSV'.

    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    Parameters
    ----------
    A : array_like
        first matrix in A-VSV' whose spectral norm is being estimated
    S : array_like
        third matrix in A-VSV' whose spectral norm is being estimated
    V : array_like
        second matrix in A-VSV' whose spectral norm is being estimated
    n_iter : int, optional
        number of iterations of the power method to conduct;
        n_iter must be a positive integer, and defaults to 20

    Returns
    -------
    float
        an estimate of the spectral norm of A-VSV' (the estimate fails
        to be accurate with exponentially low probability as n_iter
        increases; see references DS1_, DS2_, and DS3_ below)

    Examples
    --------
    >>> from fbpca import diffsnorms, eigenn
    >>> from numpy import diag
    >>> from numpy.random import uniform
    >>> from scipy.linalg import svd
    >>>
    >>> A = uniform(low=-1.0, high=1.0, size=(2, 100))
    >>> A = A.T.dot(A)
    >>> (U, s, Va) = svd(A, full_matrices=False)
    >>> A = A / s[0]
    >>>
    >>> (d, V) = eigenn(A, 2)
    >>> err = diffsnorms(A, diag(d), V)
    >>> print(err)

    This example produces a rank-2 approximation V diag(d) V' to A
    such that the columns of V are orthonormal and the entries of d
    are nonnegative and are nonincreasing.
    diffsnorms(A, diag(d), V) outputs an estimate of the spectral norm
    of A - V diag(d) V', which should be close to the machine
    precision.

    References
    ----------
    .. [DS1] Jacek Kuczynski and Henryk Wozniakowski, Estimating the
             largest eigenvalues by the power and Lanczos methods with
             a random start, SIAM Journal on Matrix Analysis and
             Applications, 13 (4): 1094-1122, 1992.
    .. [DS2] Edo Liberty, Franco Woolfe, Per-Gunnar Martinsson,
             Vladimir Rokhlin, and Mark Tygert, Randomized algorithms
             for the low-rank approximation of matrices, Proceedings of
             the National Academy of Sciences (USA), 104 (51):
             20167-20172, 2007. (See the appendix.)
    .. [DS3] Franco Woolfe, Edo Liberty, Vladimir Rokhlin, and Mark
             Tygert, A fast randomized algorithm for the approximation
             of matrices, Applied and Computational Harmonic Analysis,
             25 (3): 335-366, 2008. (See Section 3.4.)

    See also
    --------
    eigenn, eigens
    """

    m, n = np.shape(A)
    (m2, k) = V.shape
    (k2, k3) = S.shape

    assert m == n
    assert m == m2
    assert k == k2
    assert k2 == k3

    assert n_iter >= 1

    if np.isrealobj(A) and np.isrealobj(V) and np.isrealobj(S):
        isreal = True
    else:
        isreal = False

    #
    # Generate a random vector x.
    #
    if isreal:
        x = np.random.normal(size=(n, 1))
    else:
        x = np.random.normal(size=(n, 1)) + 1j * np.random.normal(size=(n, 1))

    x = x / norm(x)

    #
    # Run n_iter iterations of the power method.
    #
    for it in range(n_iter):
        #
        # Set y = (A-VSV')x.
        #
        y = mult(A, x) - V.dot(S.dot(V.conj().T.dot(x)))
        #
        # Set x = (A'-VS'V')y.
        #
        x = mult(y.conj().T, A).conj().T \
            - V.dot(S.conj().T.dot(V.conj().T.dot(y)))

        #
        # Normalize x, memorizing its Euclidean norm.
        #
        snorm = norm(x)
        if snorm == 0:
            return 0
        x = x / snorm

    snorm = math.sqrt(snorm)

    return snorm


class TestDiffsnorms(unittest.TestCase):

    def test_dense(self):

        logging.info('running TestDiffsnorms.test_dense...')
        logging.info('err =')

        for n in [100, 200]:
            for isreal in [True, False]:

                if isreal:
                    A = np.random.normal(size=(n, n))
                if not isreal:
                    A = np.random.normal(size=(n, n)) \
                        + 1j * np.random.normal(size=(n, n))

                (U, s, Va) = svd(A, full_matrices=True)
                T = (np.diag(s).dot(Va)).dot(U)
                snorm = diffsnorms(A, T, U)
                logging.info(snorm)
                self.assertTrue(snorm < .1e-10 * s[0])

    def test_sparse(self):

        logging.info('running TestDiffsnorms.test_sparse...')
        logging.info('err =')

        for n in [100, 200]:
            for isreal in [True, False]:

                if isreal:
                    A = 2 * spdiags(np.arange(n) + 1, 0, n, n)
                if not isreal:
                    A = 2 * spdiags(np.arange(n) + 1, 0, n, n) * (1 + 1j)

                A = A - spdiags(np.arange(n + 1), 1, n, n)
                A = A - spdiags(np.arange(n) + 1, -1, n, n)
                (U, s, Va) = svd(A.todense(), full_matrices=False)
                A = A / s[0]
                A = A.tocoo()

                (U, s, Va) = svd(A.todense(), full_matrices=True)
                T = (np.diag(s).dot(Va)).dot(U)
                snorm = diffsnorms(A, T, U)
                logging.info(snorm)
                self.assertTrue(snorm < .1e-10 * s[0])


def eigenn(A, k=6, n_iter=4, l=None):
    """
    Eigendecomposition of a NONNEGATIVE-DEFINITE matrix.

    Constructs a nearly optimal rank-k approximation V diag(d) V' to a
    NONNEGATIVE-DEFINITE matrix A, using n_iter normalized power
    iterations, with block size l, started with an n x l random matrix,
    when A is n x n; the reference EGN_ below explains "nearly
    optimal." k must be a positive integer <= the dimension n of A,
    n_iter must be a nonnegative integer, and l must be a positive
    integer >= k.

    The rank-k approximation V diag(d) V' comes in the form of an
    eigendecomposition -- the columns of V are orthonormal and d is a
    real vector such that its entries are nonnegative and nonincreasing.
    V is n x k and len(d) = k, when A is n x n.

    Increasing n_iter or l improves the accuracy of the approximation
    V diag(d) V'; the reference EGN_ below describes how the accuracy
    depends on n_iter and l. Please note that even n_iter=1 guarantees
    superb accuracy, whether or not there is any gap in the singular
    values of the matrix A being approximated, at least when measuring
    accuracy as the spectral norm || A - V diag(d) V' || of the matrix
    A - V diag(d) V' (relative to the spectral norm ||A|| of A).

    Notes
    -----
    THE MATRIX A MUST BE SELF-ADJOINT AND NONNEGATIVE DEFINITE.

    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    The user may ascertain the accuracy of the approximation
    V diag(d) V' to A by invoking diffsnorms(A, numpy.diag(d), V).

    Parameters
    ----------
    A : array_like, shape (n, n)
        matrix being approximated
    k : int, optional
        rank of the approximation being constructed;
        k must be a positive integer <= the dimension of A, and
        defaults to 6
    n_iter : int, optional
        number of normalized power iterations to conduct;
        n_iter must be a nonnegative integer, and defaults to 4
    l : int, optional
        block size of the normalized power iterations;
        l must be a positive integer >= k, and defaults to k+2

    Returns
    -------
    d : ndarray, shape (k,)
        vector of length k in the rank-k approximation V diag(d) V'
        to A, such that its entries are nonnegative and nonincreasing
    V : ndarray, shape (n, k)
        n x k matrix in the rank-k approximation V diag(d) V' to A,
        where A is n x n

    Examples
    --------
    >>> from fbpca import diffsnorms, eigenn
    >>> from numpy import diag
    >>> from numpy.random import uniform
    >>> from scipy.linalg import svd
    >>>
    >>> A = uniform(low=-1.0, high=1.0, size=(2, 100))
    >>> A = A.T.dot(A)
    >>> (U, s, Va) = svd(A, full_matrices=False)
    >>> A = A / s[0]
    >>>
    >>> (d, V) = eigenn(A, 2)
    >>> err = diffsnorms(A, diag(d), V)
    >>> print(err)

    This example produces a rank-2 approximation V diag(d) V' to A
    such that the columns of V are orthonormal and the entries of d
    are nonnegative and nonincreasing.
    diffsnorms(A, diag(d), V) outputs an estimate of the spectral norm
    of A - V diag(d) V', which should be close to the machine
    precision.

    References
    ----------
    .. [EGN] Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp,
             Finding structure with randomness: probabilistic
             algorithms for constructing approximate matrix
             decompositions, arXiv:0909.4061 [math.NA; math.PR], 2009
             (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).

    See also
    --------
    diffsnorms, eigens, pca
    """

    if l is None:
        l = k + 2

    m, n = np.shape(A)

    assert m == n
    assert k > 0
    assert k <= n
    assert n_iter >= 0
    assert l >= k

    if np.isrealobj(A):
        isreal = True
    else:
        isreal = False

    #
    # Check whether A is self-adjoint to nearly the machine precision.
    #
    x = np.random.uniform(low=-1.0, high=1.0, size=(n, 1))
    y = mult(A, x)
    z = mult(x.conj().T, A).conj().T
    assert (norm(y - z) <= .1e-11 * norm(y)) and \
        (norm(y - z) <= .1e-11 * norm(z))

    #
    # Eigendecompose A directly if l >= n/1.25.
    #
    if l >= n / 1.25:
        (d, V) = eigh(A.todense() if issparse(A) else A)
        #
        # Retain only the entries of d with the k greatest absolute
        # values and the corresponding columns of V.
        #
        idx = abs(d).argsort()[-k:][::-1]
        return abs(d[idx]), V[:, idx]

    #
    # Apply A to a random matrix, obtaining Q.
    #
    if isreal:
        R = np.random.uniform(low=-1.0, high=1.0, size=(n, l))
    if not isreal:
        R = np.random.uniform(low=-1.0, high=1.0, size=(n, l)) \
            + 1j * np.random.uniform(low=-1.0, high=1.0, size=(n, l))

    Q = mult(A, R)

    #
    # Form a matrix Q whose columns constitute a well-conditioned basis
    # for the columns of the earlier Q.
    #
    if n_iter == 0:

        anorm = 0
        for j in range(l):
            anorm = max(anorm, norm(Q[:, j]) / norm(R[:, j]))

        (Q, _) = qr(Q, mode='reduced')

    if n_iter > 0:

        (Q, _) = lu(Q, permute_l=True)

    #
    # Conduct normalized power iterations.
    #
    for it in range(n_iter):

        cnorm = np.zeros((l))
        for j in range(l):
            cnorm[j] = norm(Q[:, j])

        Q = mult(A, Q)

        if it + 1 < n_iter:

            (Q, _) = lu(Q, permute_l=True)

        else:

            anorm = 0
            for j in range(l):
                anorm = max(anorm, norm(Q[:, j]) / cnorm[j])

            (Q, _) = qr(Q, mode='reduced')

    #
    # Use the Nystrom method to obtain approximations to the
    # eigenvalues and eigenvectors of A (shifting A on the subspace
    # spanned by the columns of Q in order to make the shifted A be
    # positive definite). An alternative is to use the (symmetric)
    # square root in place of the Cholesky factor of the shift.
    #
    anorm = .1e-6 * anorm * math.sqrt(1. * n)
    E = mult(A, Q) + anorm * Q
    R = Q.conj().T.dot(E)
    R = (R + R.conj().T) / 2
    R = cholesky(R, lower=True)
    (E, d, V) = svd(solve(R, E.conj().T), full_matrices=False)
    V = V.conj().T
    d = d * d - anorm

    #
    # Retain only the entries of d with the k greatest absolute values
    # and the corresponding columns of V.
    #
    idx = abs(d).argsort()[-k:][::-1]
    return abs(d[idx]), V[:, idx]


class TestEigenn(unittest.TestCase):

    def test_dense(self):

        logging.info('running TestEigenn.test_dense...')

        errs = []
        err = []

        def eigenntesterrs(n, k, n_iter, isreal, l):

            if isreal:
                V = np.random.normal(size=(n, k))
            if not isreal:
                V = np.random.normal(size=(n, k)) \
                    + 1j * np.random.normal(size=(n, k))

            (V, _) = qr(V, mode='reduced')

            d0 = np.zeros((k))
            d0[0] = 1
            d0[1] = .1
            d0[2] = .01

            A = V.dot(np.diag(d0).dot(V.conj().T))
            A = (A + A.conj().T) / 2

            (d1, V1) = eigh(A)
            (d2, V2) = eigenn(A, k, n_iter, l)

            d3 = np.zeros((n))
            for ii in range(k):
                d3[ii] = d2[ii]
            d3 = sorted(d3)
            errsa = norm(d1 - d3)

            erra = diffsnorms(A, np.diag(d2), V2)

            return erra, errsa

        for n in [10, 20]:
            for k in [3, 9]:
                for n_iter in [0, 2, 1000]:
                    for isreal in [True, False]:
                        l = k + 1
                        (erra, errsa) = eigenntesterrs(n, k, n_iter, isreal, l)
                        err.append(erra)
                        errs.append(errsa)

        logging.info('errs = \n%s', np.asarray(errs))
        logging.info('err = \n%s', np.asarray(err))

        self.assertTrue(all(err[j] < .1e-10 for j in range(len(err))))

    def test_sparse(self):

        logging.info('running TestEigenn.test_sparse...')

        errs = []
        err = []
        bests = []

        def eigenntestserrs(n, k, n_iter, isreal, l):

            n2 = round(n / 2)
            assert 2 * n2 == n

            A = 2 * spdiags(np.arange(n2) + 1, 0, n, n)

            if isreal:
                A = A - spdiags(np.arange(n2 + 1), 1, n, n)
                A = A - spdiags(np.arange(n2) + 1, -1, n, n)

            if not isreal:
                A = A - 1j * spdiags(np.arange(n2 + 1), 1, n, n)
                A = A + 1j * spdiags(np.arange(n2) + 1, -1, n, n)

            A = A / diffsnorms(A, np.zeros((2, 2)), np.zeros((n, 2)))

            A = A.dot(A)
            A = A.dot(A)
            A = A.dot(A)
            A = A.dot(A)

            A = A.tocoo()

            P = np.random.permutation(n)
            A = coo_matrix((A.data, (P[A.row], P[A.col])), shape=(n, n))
            A = A.tocsr()

            (d1, V1) = eigh(A.toarray())
            (d2, V2) = eigenn(A, k, n_iter, l)

            bestsa = sorted(abs(d1))[-k - 1]

            d3 = np.zeros((n))
            for ii in range(k):
                d3[ii] = d2[ii]
            d3 = sorted(d3)
            errsa = norm(d1 - d3)

            erra = diffsnorms(A, np.diag(d2), V2)

            return erra, errsa, bestsa

        for n in [100, 200]:
            for k in [30, 90]:
                for n_iter in [2, 1000]:
                    for isreal in [True, False]:
                        l = k + 1
                        (erra, errsa, bestsa) = eigenntestserrs(n, k, n_iter,
                            isreal, l)
                        err.append(erra)
                        errs.append(errsa)
                        bests.append(bestsa)

        logging.info('errs = \n%s', np.asarray(errs))
        logging.info('err = \n%s', np.asarray(err))
        logging.info('bests = \n%s', np.asarray(bests))

        self.assertTrue(all(err[j] < max(10 * bests[j], .1e-10)
            for j in range(len(err))))


def eigens(A, k=6, n_iter=4, l=None):
    """
    Eigendecomposition of a SELF-ADJOINT matrix.

    Constructs a nearly optimal rank-k approximation V diag(d) V' to a
    SELF-ADJOINT matrix A, using n_iter normalized power iterations,
    with block size l, started with an n x l random matrix, when A is
    n x n; the reference EGS_ below explains "nearly optimal." k must
    be a positive integer <= the dimension n of A, n_iter must be a
    nonnegative integer, and l must be a positive integer >= k.

    The rank-k approximation V diag(d) V' comes in the form of an
    eigendecomposition -- the columns of V are orthonormal and d is a
    vector whose entries are real-valued and their absolute values are
    nonincreasing. V is n x k and len(d) = k, when A is n x n.

    Increasing n_iter or l improves the accuracy of the approximation
    V diag(d) V'; the reference EGS_ below describes how the accuracy
    depends on n_iter and l. Please note that even n_iter=1 guarantees
    superb accuracy, whether or not there is any gap in the singular
    values of the matrix A being approximated, at least when measuring
    accuracy as the spectral norm || A - V diag(d) V' || of the matrix
    A - V diag(d) V' (relative to the spectral norm ||A|| of A).

    Notes
    -----
    THE MATRIX A MUST BE SELF-ADJOINT.

    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    The user may ascertain the accuracy of the approximation
    V diag(d) V' to A by invoking diffsnorms(A, numpy.diag(d), V).

    Parameters
    ----------
    A : array_like, shape (n, n)
        matrix being approximated
    k : int, optional
        rank of the approximation being constructed;
        k must be a positive integer <= the dimension of A, and
        defaults to 6
    n_iter : int, optional
        number of normalized power iterations to conduct;
        n_iter must be a nonnegative integer, and defaults to 4
    l : int, optional
        block size of the normalized power iterations;
        l must be a positive integer >= k, and defaults to k+2

    Returns
    -------
    d : ndarray, shape (k,)
        vector of length k in the rank-k approximation V diag(d) V'
        to A, such that its entries are real-valued and their absolute
        values are nonincreasing
    V : ndarray, shape (n, k)
        n x k matrix in the rank-k approximation V diag(d) V' to A,
        where A is n x n

    Examples
    --------
    >>> from fbpca import diffsnorms, eigens
    >>> from numpy import diag
    >>> from numpy.random import uniform
    >>> from scipy.linalg import svd
    >>>
    >>> A = uniform(low=-1.0, high=1.0, size=(2, 100))
    >>> A = A.T.dot(A)
    >>> (U, s, Va) = svd(A, full_matrices=False)
    >>> A = A / s[0]
    >>>
    >>> (d, V) = eigens(A, 2)
    >>> err = diffsnorms(A, diag(d), V)
    >>> print(err)

    This example produces a rank-2 approximation V diag(d) V' to A
    such that the columns of V are orthonormal, and the entries of d
    are real-valued and their absolute values are nonincreasing.
    diffsnorms(A, diag(d), V) outputs an estimate of the spectral norm
    of A - V diag(d) V', which should be close to the machine
    precision.

    References
    ----------
    .. [EGS] Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp,
             Finding structure with randomness: probabilistic
             algorithms for constructing approximate matrix
             decompositions, arXiv:0909.4061 [math.NA; math.PR], 2009
             (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).

    See also
    --------
    diffsnorms, eigenn, pca
    """

    if l is None:
        l = k + 2

    m, n = np.shape(A)

    assert m == n
    assert k > 0
    assert k <= n
    assert n_iter >= 0
    assert l >= k

    if np.isrealobj(A):
        isreal = True
    else:
        isreal = False

    #
    # Check whether A is self-adjoint to nearly the machine precision.
    #
    x = np.random.uniform(low=-1.0, high=1.0, size=(n, 1))
    y = mult(A, x)
    z = mult(x.conj().T, A).conj().T
    assert (norm(y - z) <= .1e-11 * norm(y)) and \
        (norm(y - z) <= .1e-11 * norm(z))

    #
    # Eigendecompose A directly if l >= n/1.25.
    #
    if l >= n / 1.25:
        (d, V) = eigh(A.todense() if issparse(A) else A)
        #
        # Retain only the entries of d with the k greatest absolute
        # values and the corresponding columns of V.
        #
        idx = abs(d).argsort()[-k:][::-1]
        return d[idx], V[:, idx]

    #
    # Apply A to a random matrix, obtaining Q.
    #
    if isreal:
        Q = mult(A, np.random.uniform(low=-1.0, high=1.0, size=(n, l)))
    if not isreal:
        Q = mult(A, np.random.uniform(low=-1.0, high=1.0, size=(n, l))
            + 1j * np.random.uniform(low=-1.0, high=1.0, size=(n, l)))

    #
    # Form a matrix Q whose columns constitute a well-conditioned basis
    # for the columns of the earlier Q.
    #
    if n_iter == 0:
        (Q, _) = qr(Q, mode='reduced')
    if n_iter > 0:
        (Q, _) = lu(Q, permute_l=True)

    #
    # Conduct normalized power iterations.
    #
    for it in range(n_iter):

        Q = mult(A, Q)

        if it + 1 < n_iter:
            (Q, _) = lu(Q, permute_l=True)
        else:
            (Q, _) = qr(Q, mode='reduced')

    #
    # Eigendecompose Q'*A*Q to obtain approximations to the eigenvalues
    # and eigenvectors of A.
    #
    R = Q.conj().T.dot(mult(A, Q))
    R = (R + R.conj().T) / 2
    (d, V) = eigh(R)
    V = Q.dot(V)

    #
    # Retain only the entries of d with the k greatest absolute values
    # and the corresponding columns of V.
    #
    idx = abs(d).argsort()[-k:][::-1]
    return d[idx], V[:, idx]


class TestEigens(unittest.TestCase):

    def test_dense(self):

        logging.info('running TestEigens.test_dense...')

        errs = []
        err = []

        def eigenstesterrs(n, k, n_iter, isreal, l):

            if isreal:
                V = np.random.normal(size=(n, k))
            if not isreal:
                V = np.random.normal(size=(n, k)) \
                    + 1j * np.random.normal(size=(n, k))

            (V, _) = qr(V, mode='reduced')

            d0 = np.zeros((k))
            d0[0] = 1
            d0[1] = -.1
            d0[2] = .01

            A = V.dot(np.diag(d0).dot(V.conj().T))
            A = (A + A.conj().T) / 2

            (d1, V1) = eigh(A)
            (d2, V2) = eigens(A, k, n_iter, l)

            d3 = np.zeros((n))
            for ii in range(k):
                d3[ii] = d2[ii]
            d3 = sorted(d3)
            errsa = norm(d1 - d3)

            erra = diffsnorms(A, np.diag(d2), V2)

            return erra, errsa

        for n in [10, 20]:
            for k in [3, 9]:
                for n_iter in [0, 2, 1000]:
                    for isreal in [True, False]:
                        l = k + 1
                        (erra, errsa) = eigenstesterrs(n, k, n_iter, isreal, l)
                        err.append(erra)
                        errs.append(errsa)

        logging.info('errs = \n%s', np.asarray(errs))
        logging.info('err = \n%s', np.asarray(err))

        self.assertTrue(all(err[j] < .1e-10 for j in range(len(err))))

    def test_sparse(self):

        logging.info('running TestEigens.test_sparse...')

        errs = []
        err = []
        bests = []

        def eigenstestserrs(n, k, n_iter, isreal, l):

            n2 = round(n / 2)
            assert 2 * n2 == n

            A = 2 * spdiags(np.arange(n2) + 1, 0, n2, n2)

            if isreal:
                A = A - spdiags(np.arange(n2 + 1), 1, n2, n2)
                A = A - spdiags(np.arange(n2) + 1, -1, n2, n2)

            if not isreal:
                A = A - 1j * spdiags(np.arange(n2 + 1), 1, n2, n2)
                A = A + 1j * spdiags(np.arange(n2) + 1, -1, n2, n2)

            A = A / diffsnorms(A, np.zeros((2, 2)), np.zeros((n2, 2)))

            A = A.dot(A)
            A = A.dot(A)
            A = A.dot(A)
            A = A.dot(A)

            A = A.tocoo()

            datae = np.concatenate([A.data, A.data])
            rowe = np.concatenate([A.row + n2, A.row])
            cole = np.concatenate([A.col, A.col + n2])
            A = coo_matrix((datae, (rowe, cole)), shape=(n, n))

            P = np.random.permutation(n)
            A = coo_matrix((A.data, (P[A.row], P[A.col])), shape=(n, n))
            A = A.tocsc()

            (d1, V1) = eigh(A.toarray())
            (d2, V2) = eigens(A, k, n_iter, l)

            bestsa = sorted(abs(d1))[-k - 1]

            d3 = np.zeros((n))
            for ii in range(k):
                d3[ii] = d2[ii]
            d3 = sorted(d3)
            errsa = norm(d1 - d3)

            erra = diffsnorms(A, np.diag(d2), V2)

            return erra, errsa, bestsa

        for n in [100, 200]:
            for k in [30, 90]:
                for n_iter in [2, 1000]:
                    for isreal in [True, False]:
                        l = k + 1
                        (erra, errsa, bestsa) = eigenstestserrs(n, k, n_iter,
                            isreal, l)
                        err.append(erra)
                        errs.append(errsa)
                        bests.append(bestsa)

        logging.info('errs = \n%s', np.asarray(errs))
        logging.info('err = \n%s', np.asarray(err))
        logging.info('bests = \n%s', np.asarray(bests))

        self.assertTrue(all(err[j] < max(10 * bests[j], .1e-10)
            for j in range(len(err))))


def pca(A, k=6, raw=False, n_iter=2, l=None):
    """
    Principal component analysis.

    Constructs a nearly optimal rank-k approximation U diag(s) Va to A,
    centering the columns of A first when raw is False, using n_iter
    normalized power iterations, with block size l, started with a
    min(m,n) x l random matrix, when A is m x n; the reference PCA_
    below explains "nearly optimal." k must be a positive integer <=
    the smaller dimension of A, n_iter must be a nonnegative integer,
    and l must be a positive integer >= k.

    The rank-k approximation U diag(s) Va comes in the form of a
    singular value decomposition (SVD) -- the columns of U are
    orthonormal, as are the rows of Va, and the entries of s are all
    nonnegative and nonincreasing. U is m x k, Va is k x n, and
    len(s)=k, when A is m x n.

    Increasing n_iter or l improves the accuracy of the approximation
    U diag(s) Va; the reference PCA_ below describes how the accuracy
    depends on n_iter and l. Please note that even n_iter=1 guarantees
    superb accuracy, whether or not there is any gap in the singular
    values of the matrix A being approximated, at least when measuring
    accuracy as the spectral norm || A - U diag(s) Va || of the matrix
    A - U diag(s) Va (relative to the spectral norm ||A|| of A, and
    accounting for centering when raw is False).

    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    The user may ascertain the accuracy of the approximation
    U diag(s) Va to A by invoking diffsnorm(A, U, s, Va), when raw is
    True. The user may ascertain the accuracy of the approximation
    U diag(s) Va to C(A), where C(A) refers to A after centering its
    columns, by invoking diffsnormc(A, U, s, Va), when raw is False.

    Parameters
    ----------
    A : array_like, shape (m, n)
        matrix being approximated
    k : int, optional
        rank of the approximation being constructed;
        k must be a positive integer <= the smaller dimension of A,
        and defaults to 6
    raw : bool, optional
        centers A when raw is False but does not when raw is True;
        raw must be a Boolean and defaults to False
    n_iter : int, optional
        number of normalized power iterations to conduct;
        n_iter must be a nonnegative integer, and defaults to 2
    l : int, optional
        block size of the normalized power iterations;
        l must be a positive integer >= k, and defaults to k+2

    Returns
    -------
    U : ndarray, shape (m, k)
        m x k matrix in the rank-k approximation U diag(s) Va to A or
        C(A), where A is m x n, and C(A) refers to A after centering
        its columns; the columns of U are orthonormal
    s : ndarray, shape (k,)
        vector of length k in the rank-k approximation U diag(s) Va to
        A or C(A), where A is m x n, and C(A) refers to A after
        centering its columns; the entries of s are all nonnegative and
        nonincreasing
    Va : ndarray, shape (k, n)
        k x n matrix in the rank-k approximation U diag(s) Va to A or
        C(A), where A is m x n, and C(A) refers to A after centering
        its columns; the rows of Va are orthonormal

    Examples
    --------
    >>> from fbpca import diffsnorm, pca
    >>> from numpy.random import uniform
    >>> from scipy.linalg import svd
    >>>
    >>> A = uniform(low=-1.0, high=1.0, size=(100, 2))
    >>> A = A.dot(uniform(low=-1.0, high=1.0, size=(2, 100)))
    >>> (U, s, Va) = svd(A, full_matrices=False)
    >>> A = A / s[0]
    >>>
    >>> (U, s, Va) = pca(A, 2, True)
    >>> err = diffsnorm(A, U, s, Va)
    >>> print(err)

    This example produces a rank-2 approximation U diag(s) Va to A such
    that the columns of U are orthonormal, as are the rows of Va, and
    the entries of s are all nonnegative and are nonincreasing.
    diffsnorm(A, U, s, Va) outputs an estimate of the spectral norm of
    A - U diag(s) Va, which should be close to the machine precision.

    References
    ----------
    .. [PCA] Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp,
             Finding structure with randomness: probabilistic
             algorithms for constructing approximate matrix
             decompositions, arXiv:0909.4061 [math.NA; math.PR], 2009
             (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).

    See also
    --------
    diffsnorm, diffsnormc, eigens, eigenn
    """

    if l is None:
        l = k + 2

    m, n = np.shape(A)

    assert k > 0
    assert k <= min(m, n)
    assert n_iter >= 0
    assert l >= k

    if np.isrealobj(A):
        isreal = True
    else:
        isreal = False

    if raw:

        #
        # SVD A directly if l >= m/1.25 or l >= n/1.25.
        #
        if l >= m / 1.25 or l >= n / 1.25:
            (U, s, Va) = svd(A.todense() if issparse(A) else A,
                full_matrices=False)
            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]

        if m >= n:

            #
            # Apply A to a random matrix, obtaining Q.
            #
            if isreal:
                Q = mult(A, np.random.uniform(low=-1.0, high=1.0, size=(n, l)))
            if not isreal:
                Q = mult(A, np.random.uniform(low=-1.0, high=1.0, size=(n, l))
                    + 1j * np.random.uniform(low=-1.0, high=1.0, size=(n, l)))
            #
            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            #
            
            if n_iter == 0:
                (Q, _) = qr(Q, mode='reduced')
            if n_iter > 0:
                (Q, _) = lu(Q, permute_l=True)
                
            #
            # Conduct normalized power iterations.
            #
            for it in range(n_iter):
                Q = mult(Q.conj().T, A).conj().T

                (Q, _) = lu(Q, permute_l=True)

                Q = mult(A, Q)
                if it + 1 < n_iter:
                    (Q, _) = lu(Q, permute_l=True)
                else:
                    (Q, _) = qr(Q, mode='reduced')

            #
            # SVD Q'*A to obtain approximations to the singular values
            # and right singular vectors of A; adjust the left singular
            # vectors of Q'*A to approximate the left singular vectors
            # of A.
            #
            QA = mult(Q.conj().T, A)
            (R, s, Va) = svd(QA, full_matrices=False)
            U = Q.dot(R)

            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]

        if m < n:

            #
            # Apply A' to a random matrix, obtaining Q.
            #
            if isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(l, m))
            if not isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(l, m)) \
                    + 1j * np.random.uniform(low=-1.0, high=1.0, size=(l, m))

            Q = mult(R, A).conj().T

            #
            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            #
            if n_iter == 0:
                (Q, _) = qr(Q, mode='reduced')
            if n_iter > 0:
                (Q, _) = lu(Q, permute_l=True)

            #
            # Conduct normalized power iterations.
            #
            for it in range(n_iter):

                Q = mult(A, Q)
                (Q, _) = lu(Q, permute_l=True)

                Q = mult(Q.conj().T, A).conj().T

                if it + 1 < n_iter:
                    (Q, _) = lu(Q, permute_l=True)
                else:
                    (Q, _) = qr(Q, mode='reduced')

            #
            # SVD A*Q to obtain approximations to the singular values
            # and left singular vectors of A; adjust the right singular
            # vectors of A*Q to approximate the right singular vectors
            # of A.
            #
            (U, s, Ra) = svd(mult(A, Q), full_matrices=False)
            Va = Ra.dot(Q.conj().T)

            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]

    if not raw:

        #
        # Calculate the average of the entries in every column.
        #
        c = A.sum(axis=0) / m
        c = c.reshape((1, n))

        #
        # SVD the centered A directly if l >= m/1.25 or l >= n/1.25.
        #
        if l >= m / 1.25 or l >= n / 1.25:
            (U, s, Va) = svd((A.todense() if issparse(A)
                else A) - np.ones((m, 1)).dot(c), full_matrices=False)
            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]

        if m >= n:

            #
            # Apply the centered A to a random matrix, obtaining Q.
            #
            if isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(n, l))
            if not isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(n, l)) \
                    + 1j * np.random.uniform(low=-1.0, high=1.0, size=(n, l))

            Q = mult(A, R) - np.ones((m, 1)).dot(c.dot(R))

            #
            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            #
            if n_iter == 0:
                (Q, _) = qr(Q, mode='reduced')
            if n_iter > 0:
                (Q, _) = lu(Q, permute_l=True)

            #
            # Conduct normalized power iterations.
            #
            for it in range(n_iter):

                Q = (mult(Q.conj().T, A)
                    - (Q.conj().T.dot(np.ones((m, 1)))).dot(c)).conj().T
                (Q, _) = lu(Q, permute_l=True)

                Q = mult(A, Q) - np.ones((m, 1)).dot(c.dot(Q))

                if it + 1 < n_iter:
                    (Q, _) = lu(Q, permute_l=True)
                else:
                    (Q, _) = qr(Q, mode='reduced')

            #
            # SVD Q' applied to the centered A to obtain
            # approximations to the singular values and right singular
            # vectors of the centered A; adjust the left singular
            # vectors to approximate the left singular vectors of the
            # centered A.
            #
            QA = mult(Q.conj().T, A) \
                - (Q.conj().T.dot(np.ones((m, 1)))).dot(c)
            (R, s, Va) = svd(QA, full_matrices=False)
            U = Q.dot(R)

            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]

        if m < n:

            #
            # Apply the adjoint of the centered A to a random matrix,
            # obtaining Q.
            #
            
            if isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(l, m))
            if not isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(l, m)) \
                    + 1j * np.random.uniform(low=-1.0, high=1.0, size=(l, m))

            Q = (mult(R, A) - (R.dot(np.ones((m, 1)))).dot(c)).conj().T

            #
            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            #
            if n_iter == 0:
                (Q, _) = qr(Q, mode='reduced')
            if n_iter > 0:
                (Q, _) = lu(Q, permute_l=True)

            #
            # Conduct normalized power iterations.
            #
            
            for it in range(n_iter):

                Q = mult(A, Q) - np.ones((m, 1)).dot(c.dot(Q))
                (Q, _) = lu(Q, permute_l=True)

                Q = (mult(Q.conj().T, A)
                    - (Q.conj().T.dot(np.ones((m, 1)))).dot(c)).conj().T

                if it + 1 < n_iter:
                    (Q, _) = lu(Q, permute_l=True)
                else:
                    (Q, _) = qr(Q, mode='reduced')

            #
            # SVD the centered A applied to Q to obtain approximations
            # to the singular values and left singular vectors of the
            # centered A; adjust the right singular vectors to
            # approximate the right singular vectors of the centered A.
            #
            
            (U, s, Ra) = svd(mult(A, Q) - np.ones((m, 1)).dot(c.dot(Q)),
                full_matrices=False)
            Va = Ra.dot(Q.conj().T)

            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            
            return U[:, :k], s[:k], Va[:k, :]

def fp_rsvd_nondestructive(A, probability=0.9995, eps=1e-6, n_iter=2, truncate=0, l=None, random_type='Gaussian', random_seed = 0, norm_A0=None):
    """
    Fixed precision randomized SVD algorithm for large data sets. Can be used for POD, PCA etc.
    Computes an almost optimal low-rank approximation of the input matrix A = U diag(s) V, for unknown k.
    Updates A during the computations, with an option to create a deep copy.
    
    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    Parameters
    ----------
    A  : array_like, shape (m, n)
        matrix being approximated
    
    probability: float, optional
        Probability with which the threshold eps will be fulfilled (cf. Halko et al. 2011)
    
    eps: float, optional
         maximum permissible error ||A - QQ*A|| for the randomized approximation.
         Defaults to 1e-6
                  
    n_iter : int, optional
         number of normalized power iterations to conduct;
         n_iter must be a nonnegative integer, and defaults to 2
         
    sample_growth : int, optional
         - exponential growth if set to 0
         - heuristic-boosted strategy if set to 1
         - linear growth otherwise.
        
    truncate : integer, optional
         - No truncation if truncate < 0
         - Fixed precision truncation if truncate == 0 (default).
         - Fixed rank truncation to k = min(truncate, number of columns in U)
                            
    random_seed: seed for the random number generator, to achieve deterministic results
    
    random_type: type of the random number source:
         - 'Gaussian': a normal distribution with zero mean, and a standard deviation of 1.
         - 'Uniform': a uniform distribution between -1 and 1
                      Note that the implemented probabilistic error estimator only holds for Gaussian random variables.
                                                       
    l: number of samples (default/if None: l = min(max(int(n/20), 1), 4, n))
    
    normalize_A: Boolean, flag for normalization of the input matrix A by its Frobenius norm. Either True or False (default). A deep copy of A is normalized.
    
    norm_A0: Frobenius norm of A
    
    Outputs
    -------
    U : ndarray, shape (m, k)
        m x k matrix in the rank-k approximation U diag(s) Va to A or
        C(A), where A is m x n, and C(A) refers to A after centering
        its columns; the columns of U are orthonormal
    s : ndarray, shape (k,)
        vector of length k in the rank-k approximation U diag(s) Va to
        A or C(A), where A is m x n, and C(A) refers to A after
        centering its columns; the entries of s are all nonnegative and
        nonincreasing
    Va : ndarray, shape (k, n)
        k x n matrix in the rank-k approximation U diag(s) Va to A or
        C(A), where A is m x n, and C(A) refers to A after centering
        its columns; the rows of Va are orthonormal


    References
    ----------
    - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp,
             Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions 
             SIAM Review 53(2), pp. 217-288, 2011. 
             
    - Facebook Inc: Fast Randomized SVD (FBPCA algorithm)
             cf. https://github.com/facebook/fbpca
             cf. https://research.fb.com/fast-randomized-svd/
            
    
    Implementation by C. Bach (c.bach@tum.de)
    """
    
    assert n_iter >= 0
    assert eps < 32000
    
    m, n = np.shape(A)

    # safety check to account for the case that A contains only zeros
    if not np.any(A):
        print('Warning: cannot perform SVD of a matrix which contains only zero entries.')
        return(np.zeros((m,1)), np.asarray(0), np.zeros(1,n))
    
    # reset the seed of the pseudo random number generator
    np.random.seed(random_seed)
    random_type = random_type.lower()
    
    deepcopy_made = False
    if norm_A0 == None:
        norm_A0 = np.linalg.norm(A, ord='fro')

    if norm_A0 != 1.0:
        # Normalize with its Frobenius norm
        A = deepcopy(A)
        A = A / norm_A0
        norm_A0 = 1
        deepcopy_made = True
        
    # Transpose the matrix if dimensions are wrong.
    T_flag   = False
    if m < n:
        A = A.T
        m, n = np.shape(A)
        T_flag = True
        
    # Free the memory from any remaining garbage.
    gc.collect()
    
    """
    =====================================================================================================
    Carry out the actual computations.        
    =====================================================================================================
    """
    
    B = None;
    
    if l is None:
        l = min(max(int(n/20), 1), 4, n)

    # initialize error array
    Q = np.empty((m,0))
    it_count = 0

    # Estimate required number of random samples    
    r = int(np.ceil(-np.log((1-probability)/n)))
    #print('Probability %f yielded r = %d' % (probability, r))
     
    accuracy_criterion = eps / (10 * np.sqrt(2/np.pi))
    #print('accuracy_criterion = %.18f' % accuracy_criterion)
    
    if random_type == 'uniform':
        Omega = np.random.uniform(low=-1.0, high=1.0, size=(n, l))
    elif random_type == 'gaussian' or random_type == 'normal':
        Omega = np.random.normal(size=(n, l))
        
    while(True):
        Q_new = np.dot(A, Omega)
        if n_iter == 0:
            Q_new, _ = qr(Q_new, mode='reduced')
            if it_count > 0:
                Q_new = Q_new - np.dot(Q,np.dot(Q.T,Q_new))
        elif n_iter > 0:
            Q_new, _ = lu(Q_new, permute_l=True) #qr(Q_new, mode='reduced')
            if it_count > 0:
                Q_new = Q_new - np.dot(Q,np.dot(Q.T,Q_new))

            for i in range(n_iter):
                Q_new = np.dot(A, lu(np.dot(Q_new.conj().T,A).conj().T, permute_l=True)[0])
                
                if it_count > 0:
                    Q_new = Q_new - np.dot(Q,np.dot(Q.T,Q_new))

                if i+1 < n_iter:
                    Q_new, _ = lu(Q_new, permute_l = True)
                
            Q_new, _ = qr(Q_new, mode='reduced')
        
        B_new = np.dot(Q_new.conj().T,A) # dimension (l x n)
        Q = np.hstack((Q, Q_new))

        if B is None:
            B = B_new
        else:
            B = np.vstack((B, B_new))
        
        if random_type == 'uniform':
            R = np.random.uniform(low=-1.0, high=1.0, size=(n, r))
        elif random_type == 'gaussian' or random_type == 'normal':
            R = np.random.normal(size=(n, r))
            
        R = A.dot(R)
        R = R - np.dot(Q, np.dot(Q.conj().T, R))
        
        max_error = np.max([np.linalg.norm(R[:,i]) for i in range(r)])
        #print('max error = %.18f' % max_error)
        #print('||Q|| = %.18f \t\t\t ||B|| = %.18f' % (np.linalg.norm(Q), np.linalg.norm(B)))
        if max_error <= accuracy_criterion:
            break
        
        Omega = Omega - np.dot(B.T, np.dot(B, Omega))
        it_count += 1
        
    # Free the memory from any remaining garbage.
    gc.collect()
    
    # Orthonormalize Q
    Q, _ = qr(Q, mode='reduced')
        
    (R, s, Va) = svd(B, full_matrices=False)    
    U = Q.dot(R)

    # Truncate the SVD.
    # Retain only the leftmost k columns of U, the uppermost k rows of Va, and the first k entries of s.
    if truncate==0: # fixed precision for stage A
        k = 0
        for singular_value in s:
            k += 1
            if singular_value / s[0] < eps:
                break

        U = U[:,0:k]
        Va = Va[0:k,:]
        s = s[0:k]
    elif truncate > 0: # fixed rank for stage B
        k = min(truncate, np.shape(U)[1])
        U = U[:,0:k]
        Va = Va[0:k,:]
        s = s[0:k]

    if T_flag:
        U = Va.T
        Va = U.T
        if not deepcopy_made:     # if normalize_A, then a deep copy of A has already been made.
            A = A.T

    return U, s, Va

def accuracy_enhanced_r5bc(A, eps=1e-6, norm_type=-1, n_iter=2, l=None, normalize_A=False, random_type='uniform', random_seed = 0, deepcopy_A=True, sample_growth=0, norm_A0 = None):
    """
    Raw randomized rank-revealing reduced basis compucation algorithm for large data sets.
    Only computes stage A of the two-stage framework, and returns the computed range-approximator of A such that the error tolerance ||A - QQ*A|| < eps holds.
    Updates A during the computations, with an option to create a deep copy.
    
    This scheme is accuracy enhanced using an updating technique for the random matrix.
    
    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    Parameters
    ----------
    A  : array_like, shape (m, n)
        matrix being approximated
        
    eps: float, optional
         maximum permissible error ||A - QQ*A|| for the randomized approximation.
         Defaults to 1e-6
    
    norm_type: integer, optional
          denotes the matrix norm which is used for the error computation.
         -1: Frobenius norm (default)
          2: 2-norm / spectral norm
         
    n_iter : int, optional
         number of normalized power iterations to conduct;
         n_iter must be a nonnegative integer, and defaults to 2
         
    sample_growth : int, optional
         - exponential growth if set to 0
         - heuristic-boosted strategy if set to 1
         - linear growth otherwise.
                                     
    random_seed: seed for the random number generator, to achieve deterministic results
    
    random_type: type of the random number source:
         - 'Uniform': a uniform distribution between -1 and 1
         - 'Gaussian': a normal distribution with zero mean, and a standard deviation of 1.
                                                   
    deepcopy_A: Boolean. Specifies whether the matrix A which is given as an input argument can be modified.
                If not, a deepcopy of A is created before running the adaptive updating algorithm.
                
    l:          initial number of samples (default/if None: l = min(max(int(n/20), 1), 4, n))
    
    normalize_A: Boolean, flag for normalization of the input matrix A by its Frobenius norm. Either True or False (default). A deep copy of A is normalized.
    
    norm_A0: can be provided to skip the compuation of the norm of A. If use_s0_ratio == False (default), the norm is interpreted as the Frobenius norm.
             If use_s0_ratio == True, the norm is interpreted as the 2-norm.    
    
    Outputs
    -------
    Q : ndarray, shape (m, k)
        m x k matrix in the rank-k approximation, such that ||A - QQ*A|| <= eps
        The columns of U are an orthonormal basis to A.


    References
    ----------
    - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp,
             Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions 
             SIAM Review 53(2), pp. 217-288, 2011. 
             
    - Facebook Inc: Fast Randomized SVD (FBPCA algorithm)
             cf. https://github.com/facebook/fbpca
             cf. https://research.fb.com/fast-randomized-svd/
            
    
    Implementation by C. Bach (c.bach@tum.de)
    """
    
    assert n_iter >= 0
    assert eps < 32000
    
    m, n = np.shape(A)

    # safety check to account for the case that A contains only zeros
    if not np.any(A):
        print('Warning: cannot perform SVD of a matrix which contains only zero entries.')
        return(np.zeros((m,1)), np.asarray(0), np.zeros(1,n))
    
    # reset the seed of the pseudo random number generator
    np.random.seed(random_seed)
    random_type = random_type.lower()
    
    if deepcopy_A:
        A = deepcopy(A)

    if norm_A0 == None:
        if norm_type == 2:
            norm_A0 = np.linalg.norm(A, ord='2')
        else:
            norm_A0 = np.linalg.norm(A, ord='fro')

    if normalize_A:
        # Normalize with its Frobenius norm
        A = A / norm_A0
        norm_A0 = 1

    if random_type=='srft':
        A = np.asarray(A, dtype=np.complex)

    # Free the memory from any remaining garbage.
    gc.collect()
    
    """
    =====================================================================================================
    Carry out the actual computations.        
    =====================================================================================================
    """
    
    if l is None:
        l = min(max(int(n/20), 1), 4, n)
                
    # initialize error array
    last_errors = 32000 * np.ones((2,))
    Q = np.empty((m,0))
    it_count = 0
    
    B = None; Omega = np.empty((n,0))
    while(last_errors[1] > eps):
        #print('l = %d | k = %d | accuracy = %1.12f | time = %s' % (l, np.shape(Q)[1], last_errors[1], str(1000*(time.time() - t0))))
        
        # Update the next sample size unless a constant l is used (linear growth)
        if sample_growth == 1:
            if it_count < 2:
                l += l
            else:
                l = max(min(2*l, int(np.ceil(l * np.log(eps / last_errors[1]) / np.log(last_errors[1] / last_errors[0])))), 4)
        elif sample_growth == 0:
            l += l
        
        l_omega = np.shape(Omega)[1]
        if random_type == 'uniform':
            if l <= l_omega:
                Omega = Omega[:,0:l]
            else:
                Omega_new = np.random.uniform(low=-1.0, high=1.0, size=(n, l-l_omega))
                Omega = np.hstack((Omega, Omega_new))

            #Omega = np.random.uniform(low=-1.0, high=1.0, size=(n, l))
            '''
            Omega_new = np.random.uniform(low=-1.0, high=1.0, size=(n, l-np.shape(Omega)[1]))
            Omega = np.hstack((Omega, Omega_new))
            '''
            if B is not None:
                Omega = Omega - np.dot(pinv(B), np.dot(B, Omega))
            
            Q_new = np.dot(A, Omega)
            
        elif random_type == 'gaussian' or random_type == 'normal':
            if l <= l_omega:
                Omega = Omega[:,0:l]
            else:
                Omega_new = np.random.normal(size=(n, l-l_omega))
                Omega = np.hstack((Omega, Omega_new))

            #Omega = np.random.normal(size=(n, l))
            '''
            Omega_new = np.random.normal(size=(n, l-np.shape(Omega)[1]))
            Omega = np.hstack((Omega, Omega_new))
            '''
            if B is not None:
                Omega = Omega - np.dot(pinv(B), np.dot(B, Omega))
            
            Q_new = np.dot(A, Omega)

        if n_iter == 0:
            Q_new, _ = qr(Q_new, mode='reduced')
        elif n_iter > 0:
            Q_new, _ = lu(Q_new, permute_l=True) #qr(Q_new, mode='reduced')
        
            for i in range(n_iter):
                #Q_new, _ = lu(np.dot(Q_new.conj().T,A).conj().T, permute_l=True)
                #Q_new = np.dot(A, Q_new)
                Q_new = np.dot(A, lu(np.dot(Q_new.conj().T,A).conj().T, permute_l=True)[0])
                
                if i+1 < n_iter:
                    Q_new, _ = lu(Q_new, permute_l = True)
                
            Q_new, _ = qr(Q_new, mode='reduced')
            
        B_new = np.dot(Q_new.conj().T,A)
        Q = np.hstack((Q, Q_new))

        if B is None:
            B = B_new
        else:
            B = np.vstack((B, B_new))
            
        A -= np.dot(Q_new, B_new)

        last_errors[0] = last_errors[1]
        if norm_type==2:
            last_errors[1] = np.linalg.norm(A,ord='2') / norm_A0
        else:
            last_errors[1] = np.linalg.norm(A,ord='fro') / norm_A0
        
        it_count += 1
        
    # Free the memory from any remaining garbage.
    gc.collect()
    
    # Orthonormalize Q
    Q, _ = qr(Q, mode='reduced')
        
    return Q
    
def r5bc(A, eps=1e-6, norm_type=-1, n_iter=2, l=None, normalize_A=False, random_type='uniform', random_seed = 0, deepcopy_A=True, sample_growth=0, norm_A0 = None):
    """
    Raw randomized rank-revealing reduced basis compucation algorithm for large data sets.
    Only computes stage A of the two-stage framework, and returns the computed range-approximator of A such that the error tolerance ||A - QQ*A|| < eps holds.
    Updates A during the computations, with an option to create a deep copy.
    
    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    Parameters
    ----------
    A  : array_like, shape (m, n)
        matrix being approximated
        
    eps: float, optional
         maximum permissible error ||A - QQ*A|| for the randomized approximation.
         Defaults to 1e-6
    
    norm_type: integer, optional
          denotes the matrix norm which is used for the error computation.
         -1: Frobenius norm (default)
          2: 2-norm / spectral norm
         
    n_iter : int, optional
         number of normalized power iterations to conduct;
         n_iter must be a nonnegative integer, and defaults to 2
         
    sample_growth : int, optional
         - exponential growth if set to 0
         - heuristic-boosted strategy if set to 1
         - linear growth otherwise.
                                     
    random_seed: seed for the random number generator, to achieve deterministic results
    
    random_type: type of the random number source:
         - 'Uniform': a uniform distribution between -1 and 1
         - 'Gaussian': a normal distribution with zero mean, and a standard deviation of 1.
          
         - Random structured matrices.
              - 'SRHadT': a subsampled random Hadamard transform (real data)
              - 'SRFT': a subsampled random Fourier transform (complex datatype)
              - 'SRHT': a subsampled random Hartley transform (real data)
              - 'SCRT1': a subsampled random discrete cosine transform, type I (real data)
              - 'SCRT2': a subsampled random discrete cosine transform, type II (real data)
              - 'SCRT3': a subsampled random discrete cosine transform, type III (real data)
              - 'SCRT4': a subsampled random discrete cosine transform, type IV (real data)
              
           Note: efficient O(m n log(k)) multiplication of A and the random matrix NOT YET IMPLEMENTED.
           For details, see: Woolfe, F., Liberty, E., Rokhlin, V., and Tygert, M. (2008). A fast randomized algorithm for the approximation of matrices.
                             Applied and Computational Harmonic Analysis, 25(3), 335-366. http://doi.org/10.1016/j.acha.2007.12.002
                             
                             Tropp, J. A. (2011). Improved Analysis of the Subsampled Randomized Hadamard Transform. Advances in Adaptive Data Analysis, 03(01n02), 115. 
                             http://doi.org/10.1142/S1793536911000787
                             
                             Lu, Y., Dhillon, P. S., Foster, D., & Ungar, L. (2013). Faster Ridge Regression via the Subsampled Randomized Hadamard Transform.
                             Advances in Neural Information Processing Systems 26 (NIPS 2013), pp. 369-377.
                           
    deepcopy_A: Boolean. Specifies whether the matrix A which is given as an input argument can be modified.
                If not, a deepcopy of A is created before running the adaptive updating algorithm.
                
    l:          initial number of samples (default/if None: l = min(max(int(n/20), 1), 4, n))
    
    normalize_A: Boolean, flag for normalization of the input matrix A by its Frobenius norm. Either True or False (default). A deep copy of A is normalized.
    
    norm_A0: can be provided to skip the compuation of the norm of A. If use_s0_ratio == False (default), the norm is interpreted as the Frobenius norm.
             If use_s0_ratio == True, the norm is interpreted as the 2-norm.    
    
    Outputs
    -------
    Q : ndarray, shape (m, k)
        m x k matrix in the rank-k approximation, such that ||A - QQ*A|| <= eps
        The columns of U are an orthonormal basis to A.


    References
    ----------
    - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp,
             Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions 
             SIAM Review 53(2), pp. 217-288, 2011. 
             
    - Facebook Inc: Fast Randomized SVD (FBPCA algorithm)
             cf. https://github.com/facebook/fbpca
             cf. https://research.fb.com/fast-randomized-svd/
            
    
    Implementation by C. Bach (c.bach@tum.de)
    """
    
    assert n_iter >= 0
    assert eps < 32000
    
    print
    
    m, n = np.shape(A)

    # safety check to account for the case that A contains only zeros
    if not np.any(A):
        print('Warning: cannot perform SVD of a matrix which contains only zero entries.')
        return(np.zeros((m,1)), np.asarray(0), np.zeros(1,n))
    
    # reset the seed of the pseudo random number generator
    np.random.seed(random_seed)
    random_type = random_type.lower()
    
    if deepcopy_A:
        A = deepcopy(A)

    if norm_A0 == None:
        if norm_type == 2:
            norm_A0 = np.linalg.norm(A, ord='2')
        else:
            norm_A0 = np.linalg.norm(A, ord='fro')

    if normalize_A:
        # Normalize with its Frobenius norm
        A = A / norm_A0
        norm_A0 = 1

    if random_type=='srft':
        A = np.asarray(A, dtype=np.complex)
    
    # Free the memory from any remaining garbage.
    gc.collect()
    
    """
    =====================================================================================================
    Carry out the actual computations.        
    =====================================================================================================
    """
    
    if l is None:
        l = min(max(int(n/20), 1), 4, n)
                
    # initialize error array
    last_errors = 32000 * np.ones((2,))
    Q = np.empty((m,0))
    it_count = 0
    
    while(last_errors[1] > eps):
        #print('l = %d | k = %d | accuracy = %1.12f | time = %s' % (l, np.shape(Q)[1], last_errors[1], str(1000*(time.time() - t0))))
        
        # Update the next sample size unless a constant l is used (linear growth)
        if sample_growth == 1:
            if it_count < 2:
                l += l
            else:
                l = max(min(2*l, int(np.ceil(l * np.log(eps / last_errors[1]) / np.log(last_errors[1] / last_errors[0])))), 4)
        elif sample_growth == 0:
            l += l
        
        if random_type == 'uniform':
            Q_new = np.dot(A, np.random.uniform(low=-1.0, high=1.0, size=(n, l)))
        elif random_type == 'gaussian' or random_type == 'normal':
            Q_new = np.dot(A, np.random.normal(size=(n, l)))
        elif random_type == 'srft':
            D, F, R =  SRFT(n, l)
            Q_new = np.sqrt(n/l) * np.dot(A, D * F[:,R])
        elif random_type == 'srhadt':
            Q_new = np.sqrt(n/l) * mult_Had(A, l)
        elif random_type == 'srht':
            D, FR =  SRHT(n, l)
            Q_new = np.sqrt(n/l) * np.dot(A, D * FR)
        elif random_type == 'srct1':
            D, FR =  SRCT1(n, l)
            Q_new = np.sqrt(n/l) * np.dot(A, D * FR)
        elif random_type == 'srct2':
            D, FR =  SRCT2(n, l)
            Q_new = np.sqrt(n/l) * np.dot(A, D * FR)
        elif random_type == 'srct3':
            D, FR =  SRCT3(n, l)
            Q_new = np.sqrt(n/l) * np.dot(A, D * FR)
        elif random_type == 'srct4':
            D, FR =  SRCT4(n, l)
            Q_new = np.sqrt(n/l) * np.dot(A, D * FR)

        if n_iter == 0:
            Q_new, _ = qr(Q_new, mode='reduced')
        elif n_iter > 0:
            Q_new, _ = lu(Q_new, permute_l=True) #qr(Q_new, mode='reduced')
        
            for i in range(n_iter):
                #Q_new, _ = lu(np.dot(Q_new.conj().T,A).conj().T, permute_l=True)
                #Q_new = np.dot(A, Q_new)
                Q_new = np.dot(A, lu(np.dot(Q_new.conj().T,A).conj().T, permute_l=True)[0])
                
                if i+1 < n_iter:
                    Q_new, _ = lu(Q_new, permute_l = True)
                
            Q_new, _ = qr(Q_new, mode='reduced')
            
        B_new = np.dot(Q_new.conj().T,A)
        Q = np.hstack((Q, Q_new))
        #Q, _ = qr(np.hstack((Q, Q_new)), mode='reduced')
            
        A -= np.dot(Q_new, B_new)

        last_errors[0] = last_errors[1]
        if norm_type==2:
            last_errors[1] = np.linalg.norm(A,ord='2') / norm_A0
        else:
            last_errors[1] = np.linalg.norm(A,ord='fro') / norm_A0
        
        it_count += 1
        
    # Free the memory from any remaining garbage.
    gc.collect()
    
    # Orthonormalize Q
    Q, _ = qr(Q, mode='reduced')
        
    return Q

def accuracy_enhanced_fp_rsvd(A, eps=1e-6, n_iter=2, truncate=0, l=None, normalize_A=False, random_type='uniform', random_seed = 0, deepcopy_A=True, use_s0_ratio=False, sample_growth=0, norm_A0 = None):
    """
    Fixed precision randomized SVD algorithm for large data sets. Can be used for POD, PCA etc.
    Computes an almost optimal low-rank approximation of the input matrix A = U diag(s) V, for unknown k.
    Updates A during the computations, with an option to create a deep copy.
    
    This scheme is accuracy enhanced using an updating technique for the random matrix.
    
    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    Parameters
    ----------
    A  : array_like, shape (m, n)
        matrix being approximated
        
    eps: float, optional
         maximum permissible error ||A - QQ*A|| for the randomized approximation.
         Defaults to 1e-6
                  
    n_iter : int, optional
         number of normalized power iterations to conduct;
         n_iter must be a nonnegative integer, and defaults to 2
         
    sample_growth : int, optional
         - exponential growth if set to 0
         - heuristic-boosted strategy if set to 1
         - linear growth otherwise.
        
    truncate : integer, optional
         - No truncation if truncate < 0
         - Fixed precision truncation if truncate == 0 (default).
         - Fixed rank truncation to k = min(truncate, number of columns in U)
           
    use_s0_ratio: Boolean,
         - True: if a truncation of the final SVD is performed, the sv_threshold (if it is >= 0 and <=1) is interpreted as the ratio of the smallest retained singular value to the largest one.
         - False: the truncation is performed w.r.t the energy content of the singular vectors.
                  
    random_seed: seed for the random number generator, to achieve deterministic results
    
    random_type: type of the random number source:
         - 'Uniform': a uniform distribution between -1 and 1
         - 'Gaussian': a normal distribution with zero mean, and a standard deviation of 1.
          
           Note: efficient O(m n log(k)) multiplication of A and the random matrix NOT YET IMPLEMENTED.
           For details, see: Woolfe, F., Liberty, E., Rokhlin, V., and Tygert, M. (2008). A fast randomized algorithm for the approximation of matrices.
                             Applied and Computational Harmonic Analysis, 25(3), 335-366. http://doi.org/10.1016/j.acha.2007.12.002
                             
                             Tropp, J. A. (2011). Improved Analysis of the Subsampled Randomized Hadamard Transform. Advances in Adaptive Data Analysis, 03(01n02), 115. 
                             http://doi.org/10.1142/S1793536911000787
                             
                             Lu, Y., Dhillon, P. S., Foster, D., & Ungar, L. (2013). Faster Ridge Regression via the Subsampled Randomized Hadamard Transform.
                             Advances in Neural Information Processing Systems 26 (NIPS 2013), pp. 369-377.
                           
    deepcopy_A: Boolean. Specifies whether the matrix A which is given as an input argument can be modified.
                If not, a deepcopy of A is created before running the adaptive updating algorithm.
                
    l:          initial number of samples (default/if None: l = min(max(int(n/20), 1), 4, n))
    
    normalize_A: Boolean, flag for normalization of the input matrix A by its Frobenius norm. Either True or False (default). A deep copy of A is normalized.
    
    norm_A0: can be provided to skip the compuation of the norm of A. If use_s0_ratio == False (default), the norm is interpreted as the Frobenius norm.
             If use_s0_ratio == True, the norm is interpreted as the 2-norm.    
    
    Outputs
    -------
    U : ndarray, shape (m, k)
        m x k matrix in the rank-k approximation U diag(s) Va to A or
        C(A), where A is m x n, and C(A) refers to A after centering
        its columns; the columns of U are orthonormal
    s : ndarray, shape (k,)
        vector of length k in the rank-k approximation U diag(s) Va to
        A or C(A), where A is m x n, and C(A) refers to A after
        centering its columns; the entries of s are all nonnegative and
        nonincreasing
    Va : ndarray, shape (k, n)
        k x n matrix in the rank-k approximation U diag(s) Va to A or
        C(A), where A is m x n, and C(A) refers to A after centering
        its columns; the rows of Va are orthonormal


    References
    ----------
    - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp,
             Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions 
             SIAM Review 53(2), pp. 217-288, 2011. 
             
    - Facebook Inc: Fast Randomized SVD (FBPCA algorithm)
             cf. https://github.com/facebook/fbpca
             cf. https://research.fb.com/fast-randomized-svd/
            
    
    Implementation by C. Bach (c.bach@tum.de)
    """
    
    assert n_iter >= 0
    assert eps < 32000
    
    m, n = np.shape(A)

    # safety check to account for the case that A contains only zeros
    if not np.any(A):
        print('Warning: cannot perform SVD of a matrix which contains only zero entries.')
        return(np.zeros((m,1)), np.asarray(0), np.zeros(1,n))
    
    # reset the seed of the pseudo random number generator
    np.random.seed(random_seed)
    random_type = random_type.lower()
    
    if deepcopy_A:
        A = deepcopy(A)

    if norm_A0 == None:
        if use_s0_ratio:
            norm_A0 = np.linalg.norm(A, ord='2')
        else:
            norm_A0 = np.linalg.norm(A, ord='fro')

    if normalize_A:
        # Normalize with its Frobenius norm
        A = A / norm_A0
        norm_A0 = 1

    if random_type=='srft':
        A = np.asarray(A, dtype=np.complex)

    # Transpose the matrix if dimensions are wrong.
    T_flag   = False
    if m < n:
        A = A.T
        m, n = np.shape(A)
        T_flag = True
        
    # Free the memory from any remaining garbage.
    gc.collect()
    
    """
    =====================================================================================================
    Carry out the actual computations.        
    =====================================================================================================
    """
    
    B = None; Omega = np.empty((n,0))
    
    if l is None:
        l = min(max(int(n/20), 1), 4, n)
                
    # initialize error array
    last_errors = 32000 * np.ones((2,))
    Q = np.empty((m,0))
    it_count = 0
    
    mu = eps**2
    
    while(last_errors[1] > eps):
        #print('l = %d | k = %d | accuracy = %1.12f | time = %s' % (l, np.shape(Q)[1], last_errors[1], str(1000*(time.time() - t0))))
        
        # Update the next sample size unless a constant l is used (linear growth)
        if sample_growth == 1:
            if it_count < 2:
                l += l
            else:
                l = max(min(2*l, int(np.ceil(l * np.log(eps / last_errors[1]) / np.log(last_errors[1] / last_errors[0])))), 4)
        elif sample_growth == 0:
            l += l
            
        l_omega = np.shape(Omega)[1]
        if random_type == 'uniform':
            if l <= l_omega:
                Omega = Omega[:,0:l]
            else:
                Omega_new = np.random.uniform(low=-1.0, high=1.0, size=(n, l-l_omega))
                Omega = np.hstack((Omega, Omega_new))
                
            # Omega = np.random.uniform(low=-1.0, high=1.0, size=(n, l))
            '''
            Omega_new = np.random.uniform(low=-1.0, high=1.0, size=(n, l-np.shape(Omega)[1]))
            Omega = np.hstack((Omega, Omega_new))
            '''
            if B is not None:
                Omega = Omega - np.dot(pinv(B), np.dot(B, Omega))
                
            Q_new = np.dot(A, Omega)
        elif random_type == 'gaussian' or random_type == 'normal':
            if l <= l_omega:
                Omega = Omega[:,0:l]
            else:
                Omega_new = np.random.normal(size=(n, l-l_omega))
                Omega = np.hstack((Omega, Omega_new))

            # Omega = np.random.normal(size=(n, l))
            '''
            Omega_new = np.random.normal(size=(n, l-np.shape(Omega)[1]))
            Omega = np.hstack((Omega, Omega_new))
            '''
            if B is not None:
                Omega = Omega - np.dot(pinv(B), np.dot(B, Omega))
            
            Q_new = np.dot(A, Omega)

        if n_iter == 0:
            Q_new, _, = qr(Q_new, mode='reduced')
        elif n_iter > 0:
            Q_new, _ = lu(Q_new, permute_l=True) #qr(Q_new, mode='reduced')
        
            for i in range(n_iter):                
                #if(power_it_mode==1):
                Q_new = np.dot(A, lu(np.dot(Q_new.conj().T,A).conj().T, permute_l=True)[0])                    
                if i+1 < n_iter:
                    Q_new, _ = lu(Q_new, permute_l = True)
                
            Q_new, _ = qr(Q_new, mode='reduced')
            
        B_new = np.dot(Q_new.conj().T,A)
        Q = np.hstack((Q, Q_new))
        #Q, _ = qr(np.hstack((Q, Q_new)), mode='reduced')

        if B is None:
            B = B_new
        else:
            B = np.vstack((B, B_new))
            
        A -= np.dot(Q_new, B_new)

        last_errors[0] = last_errors[1]
        if use_s0_ratio:
            last_errors[1] = np.linalg.norm(A,ord='2') / norm_A0
        else:
            last_errors[1] = np.linalg.norm(A,ord='fro') / norm_A0
        
        it_count += 1
        
    # Free the memory from any remaining garbage.
    gc.collect()
    
    # Orthonormalize Q
    Q, _ = qr(Q, mode='reduced')
    
    (R, s, Va) = svd(B, full_matrices=False)    
    U = Q.dot(R)

    # Truncate the SVD.
    # Retain only the leftmost k columns of U, the uppermost k rows of Va, and the first k entries of s.
    if truncate==0: # fixed precision for stage A
        k = 0
        if use_s0_ratio:
            for singular_value in s:
                if singular_value / norm_A0 < eps:
                    break
                k += 1
        else:
            energies = np.flipud(np.cumsum(np.flipud(s * s)))
            energies /= (norm_A0**2)
            for k in range(0, len(energies)):
                if energies[k] <= mu:
                    break
            #k += 1

        U = U[:,0:max(1,k)]
        Va = Va[0:max(1,k),:]
        s = s[0:max(1,k)]
            
    elif truncate == 1: # fixed rank for stage B
        k = min(truncate, np.shape(U)[1])
        U = U[:,0:k]
        Va = Va[0:k,:]
        s = s[0:k]

    if T_flag:
        U = Va.conj().T
        Va = U.conj().T

    return U, s, Va

def fp_rsvd(A, eps=1e-6, n_iter=2, truncate=0, l=None, normalize_A=False, random_type='uniform', random_seed = 0, deepcopy_A=True, use_s0_ratio=False, sample_growth=0, norm_A0 = None):
    """
    Fixed precision randomized SVD algorithm for large data sets. Can be used for POD, PCA etc.
    Computes an almost optimal low-rank approximation of the input matrix A = U diag(s) V, for unknown k.
    Updates A during the computations, with an option to create a deep copy.
    
    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    Parameters
    ----------
    A  : array_like, shape (m, n)
        matrix being approximated
        
    eps: float, optional
         maximum permissible error ||A - QQ*A|| for the randomized approximation.
         Defaults to 1e-6
                  
    n_iter : int, optional
         number of normalized power iterations to conduct;
         n_iter must be a nonnegative integer, and defaults to 2
         
    sample_growth : int, optional
         - exponential growth if set to 0
         - heuristic-boosted strategy if set to 1
         - linear growth otherwise.
        
    truncate : integer, optional
         - No truncation if truncate < 0
         - Fixed precision truncation if truncate == 0 (default).
         - Fixed rank truncation to k = min(truncate, number of columns in U)
           
    use_s0_ratio: Boolean,
         - True: if a truncation of the final SVD is performed, the sv_threshold (if it is >= 0 and <=1) is interpreted as the ratio of the smallest retained singular value to the largest one.
         - False: the truncation is performed w.r.t the energy content of the singular vectors.
                  
    random_seed: seed for the random number generator, to achieve deterministic results
    
    random_type: type of the random number source:
         - 'Uniform': a uniform distribution between -1 and 1
         - 'Gaussian': a normal distribution with zero mean, and a standard deviation of 1.
          
         - Random structured matrices.
              - 'SRHadT': a subsampled random Hadamard transform (real data)
              - 'SRFT': a subsampled random Fourier transform (complex datatype)
              - 'SRHT': a subsampled random Hartley transform (real data)
              - 'SCRT1': a subsampled random discrete cosine transform, type I (real data)
              - 'SCRT2': a subsampled random discrete cosine transform, type II (real data)
              - 'SCRT3': a subsampled random discrete cosine transform, type III (real data)
              - 'SCRT4': a subsampled random discrete cosine transform, type IV (real data)
              
           Note: efficient O(m n log(k)) multiplication of A and the random matrix NOT YET IMPLEMENTED.
           For details, see: Woolfe, F., Liberty, E., Rokhlin, V., and Tygert, M. (2008). A fast randomized algorithm for the approximation of matrices.
                             Applied and Computational Harmonic Analysis, 25(3), 335-366. http://doi.org/10.1016/j.acha.2007.12.002
                             
                             Tropp, J. A. (2011). Improved Analysis of the Subsampled Randomized Hadamard Transform. Advances in Adaptive Data Analysis, 03(01n02), 115. 
                             http://doi.org/10.1142/S1793536911000787
                             
                             Lu, Y., Dhillon, P. S., Foster, D., & Ungar, L. (2013). Faster Ridge Regression via the Subsampled Randomized Hadamard Transform.
                             Advances in Neural Information Processing Systems 26 (NIPS 2013), pp. 369-377.
                           
    deepcopy_A: Boolean. Specifies whether the matrix A which is given as an input argument can be modified.
                If not, a deepcopy of A is created before running the adaptive updating algorithm.
                
    l:          initial number of samples (default/if None: l = min(max(int(n/20), 1), 4, n))
    
    normalize_A: Boolean, flag for normalization of the input matrix A by its Frobenius norm. Either True or False (default). A deep copy of A is normalized.
    
    norm_A0: can be provided to skip the compuation of the norm of A. If use_s0_ratio == False (default), the norm is interpreted as the Frobenius norm.
             If use_s0_ratio == True, the norm is interpreted as the 2-norm.    
    
    Outputs
    -------
    U : ndarray, shape (m, k)
        m x k matrix in the rank-k approximation U diag(s) Va to A or
        C(A), where A is m x n, and C(A) refers to A after centering
        its columns; the columns of U are orthonormal
    s : ndarray, shape (k,)
        vector of length k in the rank-k approximation U diag(s) Va to
        A or C(A), where A is m x n, and C(A) refers to A after
        centering its columns; the entries of s are all nonnegative and
        nonincreasing
    Va : ndarray, shape (k, n)
        k x n matrix in the rank-k approximation U diag(s) Va to A or
        C(A), where A is m x n, and C(A) refers to A after centering
        its columns; the rows of Va are orthonormal


    References
    ----------
    - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp,
             Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions 
             SIAM Review 53(2), pp. 217-288, 2011. 
             
    - Facebook Inc: Fast Randomized SVD (FBPCA algorithm)
             cf. https://github.com/facebook/fbpca
             cf. https://research.fb.com/fast-randomized-svd/
            
    
    Implementation by C. Bach (c.bach@tum.de)
    """
    
    assert n_iter >= 0
    assert eps < 32000
    
    m, n = np.shape(A)

    # safety check to account for the case that A contains only zeros
    if not np.any(A):
        print('Warning: cannot perform SVD of a matrix which contains only zero entries.')
        return(np.zeros((m,1)), np.asarray(0), np.zeros(1,n))
    
    # reset the seed of the pseudo random number generator
    np.random.seed(random_seed)
    random_type = random_type.lower()
    
    if deepcopy_A:
        A = deepcopy(A)

    if norm_A0 == None:
        if use_s0_ratio:
            norm_A0 = np.linalg.norm(A, ord='2')
        else:
            norm_A0 = np.linalg.norm(A, ord='fro')

    if normalize_A:
        # Normalize with its Frobenius norm
        A = A / norm_A0
        norm_A0 = 1

    if random_type=='srft':
        A = np.asarray(A, dtype=np.complex)

    # Transpose the matrix if dimensions are wrong.
    T_flag   = False
    if m < n:
        A = A.T
        m, n = np.shape(A)
        T_flag = True
        
    # Free the memory from any remaining garbage.
    gc.collect()
    
    """
    =====================================================================================================
    Carry out the actual computations.        
    =====================================================================================================
    """
    
    B = None;
    
    if l is None:
        l = min(max(int(n/20), 1), 4, n)
                
    # initialize error array
    last_errors = 32000 * np.ones((2,))
    Q = np.empty((m,0))
    it_count = 0
    
    mu = eps**2
    
    while(last_errors[1] > eps):
        #print('l = %d | k = %d | accuracy = %1.12f | time = %s' % (l, np.shape(Q)[1], last_errors[1], str(1000*(time.time() - t0))))
        
        # Update the next sample size unless a constant l is used (linear growth)
        if sample_growth == 1:
            if it_count < 2:
                l += l
            else:
                l = max(min(2*l, int(np.ceil(l * np.log(eps / last_errors[1]) / np.log(last_errors[1] / last_errors[0])))), 4)
        elif sample_growth == 0:
            l += l
        
        if random_type == 'uniform':
            Q_new = np.dot(A, np.random.uniform(low=-1.0, high=1.0, size=(n, l)))
        elif random_type == 'gaussian' or random_type == 'normal':
            Q_new = np.dot(A, np.random.normal(size=(n, l)))
        elif random_type == 'srft':
            D, F, R =  SRFT(n, l)
            Q_new = np.sqrt(n/l) * np.dot(A, D * F[:,R])
        elif random_type == 'srhadt':
            Q_new = np.sqrt(n/l) * mult_Had(A, l)
        elif random_type == 'srht':
            D, FR =  SRHT(n, l)
            Q_new = np.sqrt(n/l) * np.dot(A, D * FR)
        elif random_type == 'srct1':
            D, FR =  SRCT1(n, l)
            Q_new = np.sqrt(n/l) * np.dot(A, D * FR)
        elif random_type == 'srct2':
            D, FR =  SRCT2(n, l)
            Q_new = np.sqrt(n/l) * np.dot(A, D * FR)
        elif random_type == 'srct3':
            D, FR =  SRCT3(n, l)
            Q_new = np.sqrt(n/l) * np.dot(A, D * FR)
        elif random_type == 'srct4':
            D, FR =  SRCT4(n, l)
            Q_new = np.sqrt(n/l) * np.dot(A, D * FR)

        if n_iter == 0:
            Q_new, _, = qr(Q_new, mode='reduced')
        elif n_iter > 0:
            Q_new, _ = lu(Q_new, permute_l=True) #qr(Q_new, mode='reduced')
        
            for i in range(n_iter):                
                #if(power_it_mode==1):
                Q_new = np.dot(A, lu(np.dot(Q_new.conj().T,A).conj().T, permute_l=True)[0])
                #else:
                #    Q_new = np.dot(A.T, Q_new)
                #    if B is not None:
                #        Q_new = Q_new - np.dot(B.T, np.dot(B, Q_new))  # (n x l) ... Q = (m x s) 
                #    Q_new, _ = lu(Q_new, permute_l = True)
                #    Q_new = np.dot(A, Q_new)
                #    if B is not None:
                #        Q_new = Q_new - np.dot(Q, np.dot(Q.T, Q_new))  # (m x l) - (m x s) * (s x m) * (m x l)
                    
                if i+1 < n_iter:
                    Q_new, _ = lu(Q_new, permute_l = True)
                
                
            Q_new, _ = qr(Q_new, mode='reduced')
            
        B_new = np.dot(Q_new.conj().T,A)
        Q = np.hstack((Q, Q_new))
        #Q, _ = qr(np.hstack((Q, Q_new)), mode='reduced')

        if B is None:
            B = B_new
        else:
            B = np.vstack((B, B_new))
            
        A -= np.dot(Q_new, B_new)

        last_errors[0] = last_errors[1]
        if use_s0_ratio:
            last_errors[1] = np.linalg.norm(A,ord='2') / norm_A0
        else:
            last_errors[1] = np.linalg.norm(A,ord='fro') / norm_A0
        
        it_count += 1
        
    # Free the memory from any remaining garbage.
    gc.collect()
    
    # Orthonormalize Q
    Q, _ = qr(Q, mode='reduced')
    
    (R, s, Va) = svd(B, full_matrices=False)    
    U = Q.dot(R)

    # Truncate the SVD.
    # Retain only the leftmost k columns of U, the uppermost k rows of Va, and the first k entries of s.
    if truncate==0: # fixed precision for stage A
        k = 0
        if use_s0_ratio:
            for singular_value in s:
                if singular_value / norm_A0 < eps:
                    break
                k += 1
        else:
            energies = np.flipud(np.cumsum(np.flipud(s * s)))
            energies /= (norm_A0**2)
            for k in range(0, len(energies)):
                if energies[k] <= mu:
                    break
                
            #k += 1

        U = U[:,0:max(1,k)]
        Va = Va[0:max(1,k),:]
        s = s[0:max(1,k)]
            
    elif truncate == 1: # fixed rank for stage B
        k = min(truncate, np.shape(U)[1])
        U = U[:,0:k]
        Va = Va[0:k,:]
        s = s[0:k]

    if T_flag:
        U = Va.conj().T
        Va = U.conj().T

    return U, s, Va

def fp_rsvd_v2(A, eps=1e-6, n_iter=2, l=None, random_type='uniform', random_seed = 0, deepcopy_A=True, use_s0_ratio=False, sample_growth=0):
    """
    Fixed precision randomized SVD algorithm for large data sets. Can be used for POD, PCA etc.
    Computes an almost optimal low-rank approximation of the input matrix A = U diag(s) V, for unknown k.
    Updates A during the computations, with an option to create a deep copy.
    
    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    Parameters
    ----------
    A  : array_like, shape (m, n)
        matrix being approximated
        
    eps: float, optional
         maximum permissible error ||A - QQ*A|| for the randomized approximation.
         Defaults to 1e-6
                  
    n_iter : int, optional
         number of normalized power iterations to conduct;
         n_iter must be a nonnegative integer, and defaults to 2
         
    sample_growth : int, optional
         - exponential growth if set to 0
         - heuristic-boosted strategy if set to 1
         - linear growth otherwise.
                   
    use_s0_ratio: Boolean,
         - True: uses L2-norm (spectral norm)
         - False: uses Frobenius norm
                  
    random_seed: seed for the random number generator, to achieve deterministic results
    
    random_type: type of the random number source:
         - 'Uniform': a uniform distribution between -1 and 1
         - 'Gaussian': a normal distribution with zero mean, and a standard deviation of 1.
          
         - Random structured matrices.
              - 'SRHadT': a subsampled random Hadamard transform (real data)
              - 'SRFT': a subsampled random Fourier transform (complex datatype)
              - 'SRHT': a subsampled random Hartley transform (real data)
              - 'SCRT1': a subsampled random discrete cosine transform, type I (real data)
              - 'SCRT2': a subsampled random discrete cosine transform, type II (real data)
              - 'SCRT3': a subsampled random discrete cosine transform, type III (real data)
              - 'SCRT4': a subsampled random discrete cosine transform, type IV (real data)
              
           Note: efficient O(m n log(k)) multiplication of A and the random matrix NOT YET IMPLEMENTED.
           For details, see: Woolfe, F., Liberty, E., Rokhlin, V., and Tygert, M. (2008). A fast randomized algorithm for the approximation of matrices.
                             Applied and Computational Harmonic Analysis, 25(3), 335-366. http://doi.org/10.1016/j.acha.2007.12.002
                             
                             Tropp, J. A. (2011). Improved Analysis of the Subsampled Randomized Hadamard Transform. Advances in Adaptive Data Analysis, 03(01n02), 115. 
                             http://doi.org/10.1142/S1793536911000787
                             
                             Lu, Y., Dhillon, P. S., Foster, D., & Ungar, L. (2013). Faster Ridge Regression via the Subsampled Randomized Hadamard Transform.
                             Advances in Neural Information Processing Systems 26 (NIPS 2013), pp. 369-377.
                           
    deepcopy_A: Boolean. Specifies whether the matrix A which is given as an input argument can be modified.
                If not, a deepcopy of A is created before running the adaptive updating algorithm.
                
    l:          initial number of samples (default/if None: l = min(max(int(n/20), 1), 4, n))
    
    normalize: Boolean, flag for normalization of the input matrix A by its Frobenius norm. Either True or False (default)
    
    
    Outputs
    -------
    U : ndarray, shape (m, k)
        m x k matrix in the rank-k approximation U diag(s) Va to A or
        C(A), where A is m x n, and C(A) refers to A after centering
        its columns; the columns of U are orthonormal
    s : ndarray, shape (k,)
        vector of length k in the rank-k approximation U diag(s) Va to
        A or C(A), where A is m x n, and C(A) refers to A after
        centering its columns; the entries of s are all nonnegative and
        nonincreasing
    Va : ndarray, shape (k, n)
        k x n matrix in the rank-k approximation U diag(s) Va to A or
        C(A), where A is m x n, and C(A) refers to A after centering
        its columns; the rows of Va are orthonormal


    References
    ----------
    - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp,
             Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions 
             SIAM Review 53(2), pp. 217-288, 2011. 
             
    - Facebook Inc: Fast Randomized SVD (FBPCA algorithm)
             cf. https://github.com/facebook/fbpca
             cf. https://research.fb.com/fast-randomized-svd/
            
    
    Implementation by C. Bach (c.bach@tum.de)
    """
    
    assert n_iter >= 0
    assert eps < 32000
    
    m, n = np.shape(A)

    # safety check to account for the case that A contains only zeros
    if not np.any(A):
        print('Warning: cannot perform SVD of a matrix which contains only zero entries.')
        return(np.zeros((m,1)), np.asarray(0), np.zeros(1,n))
    
    # reset the seed of the pseudo random number generator
    np.random.seed(random_seed)
    random_type = random_type.lower()
    
    if deepcopy_A:
        A = deepcopy(A)

    if random_type=='srft':
        A = np.asarray(A, dtype=np.complex)

    # Transpose the matrix if dimensions are wrong.
    T_flag   = False
    if m < n:
        A = A.T
        m, n = np.shape(A)
        T_flag = True
        
    # Free the memory from any remaining garbage.
    gc.collect()
    
    """
    =====================================================================================================
    Carry out the actual computations.        
    =====================================================================================================
    """
    
    B = None;
    
    if l is None:
        l = min(max(int(n/20), 1), 4, n)
                
    # initialize error array
    last_errors = 32000 * np.ones((2,))
    Q = np.empty((m,0))
    it_count = 0
    
    if use_s0_ratio:
        norm_A0 = np.linalg.norm(A, ord='2')
    else:
        norm_A0 = np.linalg.norm(A, ord='fro')
        
    mu = eps**2
    
    while(last_errors[1] > eps):
        #print('l = %d | k = %d | accuracy = %1.12f | time = %s' % (l, np.shape(Q)[1], last_errors[1], str(1000*(time.time() - t0))))
        
        # Update the next sample size unless a constant l is used (linear growth)
        if sample_growth == 1:
            if it_count < 2:
                l += l
            else:
                l = max(min(2*l, int(np.ceil(l * np.log(eps / last_errors[1]) / np.log(last_errors[1] / last_errors[0])))), 4)
        elif sample_growth == 0:
            l += l
        
        if random_type == 'uniform':
            Q_new = np.dot(A, np.random.uniform(low=-1.0, high=1.0, size=(n, l)))
        elif random_type == 'gaussian' or random_type == 'normal':
            Q_new = np.dot(A, np.random.normal(size=(n, l)))
        elif random_type == 'srft':
            D, F, R =  SRFT(n, l)
            Q_new = np.sqrt(n/l) * np.dot(A, D * F[:,R])
        elif random_type == 'srhadt':
            Q_new = np.sqrt(n/l) * mult_Had(A, l)
        elif random_type == 'srht':
            D, FR =  SRHT(n, l)
            Q_new = np.sqrt(n/l) * np.dot(A, D * FR)
        elif random_type == 'srct1':
            D, FR =  SRCT1(n, l)
            Q_new = np.sqrt(n/l) * np.dot(A, D * FR)
        elif random_type == 'srct2':
            D, FR =  SRCT2(n, l)
            Q_new = np.sqrt(n/l) * np.dot(A, D * FR)
        elif random_type == 'srct3':
            D, FR =  SRCT3(n, l)
            Q_new = np.sqrt(n/l) * np.dot(A, D * FR)
        elif random_type == 'srct4':
            D, FR =  SRCT4(n, l)
            Q_new = np.sqrt(n/l) * np.dot(A, D * FR)

        if n_iter == 0:
            Q_new, _ = qr(Q_new, mode='reduced')
        elif n_iter > 0:
            Q_new, _ = lu(Q_new, permute_l=True) #qr(Q_new, mode='reduced')
        
            for i in range(n_iter):
                #Q_new, _ = lu(np.dot(Q_new.conj().T,A).conj().T, permute_l=True)
                #Q_new = np.dot(A, Q_new)
                Q_new = np.dot(A, lu(np.dot(Q_new.conj().T,A).conj().T, permute_l=True)[0])
                
                if i+1 < n_iter:
                    Q_new, _ = lu(Q_new, permute_l = True)
                
            Q_new, _ = qr(Q_new, mode='reduced')
            
        B_new = np.dot(Q_new.conj().T,A)
        Q = np.hstack((Q, Q_new))
        #Q, _ = qr(np.hstack((Q, Q_new)), mode='reduced')

        if B is None:
            B = B_new
        else:
            B = np.vstack((B, B_new))
            
        A -= np.dot(Q_new, B_new)
        
        last_errors[0] = last_errors[1]
        if use_s0_ratio:
            last_errors[1] = np.linalg.norm(A,ord='2') / norm_A0
        else:
            last_errors[1] = np.linalg.norm(A,ord='fro') / norm_A0

        it_count += 1
        
    # Free the memory from any remaining garbage.
    gc.collect()
    
    # Orthonormalize Q
    Q, _ = qr(Q, mode='reduced')
        
    (R, s, Va) = svd(B, full_matrices=False)    
    U = Q.dot(R)

    # Truncate the SVD.
    # Retain only the leftmost k columns of U, the uppermost k rows of Va, and the first k entries of s.
    k = 0
    if use_s0_ratio:
        for singular_value in s:
            if singular_value / norm_A0 < eps:
                break
            k += 1
    else:
        energies = np.flipud(np.cumsum(np.flipud(s * s)))
        energies /= (norm_A0**2)
        for k in range(0, len(energies)):
            if energies[k] <= mu:
                break
                
        #k += 1

    U = U[:,0:max(1,k)]
    Va = Va[0:max(1,k),:]
    s = s[0:max(1,k)]

    if T_flag:
        U = Va.T
        Va = U.T
        
    return U, s, Va

def fixed_precision_large_rsvd(A, eps=1e-6, probability = 0.9995, n_iter=2, truncate=0, l=None, variant='single Q', random_type='Gaussian', random_seed = 0, norm_A0 = None):
    """
    Fixed precision randomized SVD algorithm for large data sets. Can be used for POD, PCA etc.
    Computes an almost optimal low-rank approximation of the input matrix A = U diag(s) V, for unknown k.
    
    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    Parameters
    ----------
    A  : array_like, shape (m, n)
        matrix being approximated
        
    eps: float, optional
         maximum permissible error ||A - QQ*A|| for the randomized approximation.
         Defaults to 1e-6
         
    probability: float, optional
         probability with which this error threshold is met.
         This value is not used in the 'Adaptive updating' strategy.
         
    n_iter : int, optional
         number of normalized power iterations to conduct;
         n_iter must be a nonnegative integer, and defaults to 2
         
    truncate : integer, optional
         - No truncation if truncate < 0
         - Fixed precision truncation if truncate == 0 (default).
         - Fixed rank truncation to k = min(truncate, number of columns in U)

    variant: different variants of the algorithm. Available options:
         - 'intermediate-saving'
         - 'single Q' (default)
         
    random_seed: seed for the random number generator, to achieve deterministic results
    
    random_type: type of the random number source:
         - 'Gaussian': a normal distribution with zero mean, and a standard deviation of 1.
         - 'Uniform': a uniform distribution between -1 and 1
            Note that the implemented probabilistic error estimators are only valid for a Gaussian standard distribution.
                                                    
    l:  number of samples (default/if None: l = min(max(int(n/20), 1), 4, n))
    
    normalize_A: Boolean, flag for normalization of the input matrix A by its Frobenius norm. Either True or False (default). A deep copy of A is normalized.
    
    norm_A0: can be provided to skip the compuation of the norm of A. If use_s0_ratio == False (default), the norm is interpreted as the Frobenius norm.
             If use_s0_ratio == True, the norm is interpreted as the 2-norm.    
    
    Outputs
    -------
    U : ndarray, shape (m, k)
        m x k matrix in the rank-k approximation U diag(s) Va to A or
        C(A), where A is m x n, and C(A) refers to A after centering
        its columns; the columns of U are orthonormal
    s : ndarray, shape (k,)
        vector of length k in the rank-k approximation U diag(s) Va to
        A or C(A), where A is m x n, and C(A) refers to A after
        centering its columns; the entries of s are all nonnegative and
        nonincreasing
    Va : ndarray, shape (k, n)
        k x n matrix in the rank-k approximation U diag(s) Va to A or
        C(A), where A is m x n, and C(A) refers to A after centering
        its columns; the rows of Va are orthonormal


    References
    ----------
    - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp:
             Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions 
             SIAM Review 53(2), pp. 217-288, 2011. 
             
    - Nathan Halko, Per-Gunnar Martinsson, Yoel Shkolnisky, Mark Tygert:
             An Algorithm for the Principal Component Analysis of Large Data Sets
             SIAM Journal on Scientific Computing 33(5), pp. 2580-2594, 2011.
    
    - Facebook Inc: Fast Randomized SVD (FBPCA algorithm)
             cf. https://github.com/facebook/fbpca
             cf. https://research.fb.com/fast-randomized-svd/
            
    
    Implementation by C. Bach (c.bach@tum.de)
    """
    
    assert n_iter >= 0
    assert eps < 32000
    
    m, n = np.shape(A)

    # safety check to account for the case that A contains only zeros
    if not np.any(A):
        print('Warning: cannot perform SVD of a matrix which contains only zero entries.')
        return(np.zeros((m,1)), np.asarray(0), np.zeros(1,n))
    
    # reset the seed of the pseudo random number generator
    np.random.seed(random_seed)
    random_type = random_type.lower()
    
    # Transpose the matrix if dimensions are wrong.
    T_flag   = False
    deepcopy_made = False
    
    if m < n:
        A = A.T
        m, n = np.shape(A)
        T_flag = True
        
    if norm_A0 == None:
        norm_A0 = np.linalg.norm(A, ord='fro')

    if norm_A0 != 1.0:
        # Normalize with its Frobenius norm
        A = deepcopy(A)
        A = A / norm_A0
        norm_A0 = 1
        deepcopy_made = True

    # Free the memory from any remaining garbage.
    gc.collect()
    
    """
    =====================================================================================================
    Carry out the actual computations.        
    =====================================================================================================
    """
    
    Q = None
    B = None
    
    if l is None:
        l = min(max(int(n/20), 1), 4, n)
        
    '''
    Obtain r from the specified minimum probability. The relationship can be written as:
    probability >= 1 - min(n,m) * 10**(-r)
    '''
    r = int(np.ceil(-np.log((1-probability)/n)))
    #print('Probability %f yielded r = %d' % (probability, r))
    accuracy_criterion = eps / (10 * np.sqrt(2/np.pi))

    it_count = 0
    last_errors = 32000 * np.ones((2,))

    if random_type == 'uniform':
        Omega = np.random.uniform(low=-1.0, high=1.0, size=(n, l))
    elif random_type == 'gaussian' or random_type == 'normal':
        Omega = np.random.normal(size=(n, l))
        
    while True:
        # Update the next sample size unless a constant l is used (linear growth)            
        if variant == 'single Q':
            Q_new = np.dot(A, Omega)
        else:
            # Create the W matrix, which contains contributions from several power iterations.
            if n_iter == 0:
                W = np.empty((m,(n_iter+1)*l))
                
                W[:,0:l], _ = qr(np.dot(A, Omega), mode='reduced', pivoting=True)
                if it_count > 0:
                    W[:,0:l] = W[:,0:l] - np.dot(Q,np.dot(Q.T,W[:,0:l]))

            elif n_iter > 0:
                W = np.empty((m,(n_iter+1)*l))
                W[:,0:l], _ = lu(np.dot(A, Omega), permute_l=True)
                if it_count > 0:
                    W[:,0:l] = W[:,0:l] - np.dot(Q,np.dot(Q.T,W[:,0:l]))

        if variant == 'intermediate-saving':    
            # Orthonormalize the columns of Q
            for i in range(1, n_iter+1):
                Q_tmp = np.dot(A, lu(np.dot(W[:,(i-1)*l:i*l].conj().T,A).conj().T, permute_l=True)[0])
                #Q_tmp = np.dot(A, W[:,(i-1)*l:i*l])
                if it_count > 0:
                    Q_tmp = Q_tmp - np.dot(Q,np.dot(Q.T,Q_tmp))
                                          
                if i+2 < n_iter:
                    W[:,i*l:(i+1)*l], _ = lu(Q_tmp, permute_l = True)
                else:
                    W[:,i*l:(i+1)*l], _ = qr(Q_tmp, mode='reduced')
                    
                if it_count > 0:
                    W[:,i*l:(i+1)*l] = W[:,i*l:(i+1)*l] - np.dot(Q,np.dot(Q.T,W[:,i*l:(i+1)*l]))
                                             
            # See: 
            # "An Algorithm for the Principal Component Analysis of Large Data Sets" (2011)
            # by Halko NP, Martinsson P-G, Shkolnisky Y, Tygert M
            
            Q_new, _, _ = spqr(W, mode='economic', pivoting=True)
            Q_new = Q_new[:,0:l]
            
        elif variant == 'single Q':
            # Normalized power iterations.
            if n_iter == 0:
                Q_new, _ = qr(Q_new, mode='reduced')
                if it_count > 0:
                    Q_new = Q_new - np.dot(Q,np.dot(Q.T,Q_new))
            elif n_iter > 0:
                Q_new, _ = lu(Q_new, permute_l=True) #qr(Q_new, mode='reduced')
                if it_count > 0:
                    Q_new = Q_new - np.dot(Q,np.dot(Q.T,Q_new))
    
                for i in range(n_iter):
                    Q_new = np.dot(A, lu(np.dot(Q_new.conj().T,A).conj().T, permute_l=True)[0])
                    
                    if it_count > 0:
                        Q_new = Q_new - np.dot(Q,np.dot(Q.T,Q_new))
    
                    if i+1 < n_iter:
                        Q_new, _ = lu(Q_new, permute_l = True)
                        
                Q_new, _ = qr(Q_new, mode='reduced')
                
        # Free the memory from any remaining garbage.
        gc.collect()

        B_new = np.dot(Q_new.conj().T,A)
        if Q is None:
            Q = Q_new
        else:
            Q = np.hstack((Q, Q_new))
            #Q, _ = qr(np.hstack((Q, Q_new)), mode='reduced')
        
        if B is None:
            B = B_new
        else:
            B = np.vstack((B, B_new))

        # Draw r random vectors
        if random_type == 'uniform':
            R = np.random.uniform(low=-1.0, high=1.0, size=(n, r))
        elif random_type == 'gaussian' or random_type == 'normal':
            R = np.random.normal(size=(n, r))
        
        # Equivalent to R = (I - Q Q*)AR, but faster
        R = np.dot(A,R)
        R = R - np.dot(Q, np.dot(Q.conj().T,R))
        
        last_errors[0] = last_errors[1]
        last_errors[1] = np.max([np.linalg.norm(R[:,i]) for i in range(r)])
        if last_errors[1] < accuracy_criterion:
            break
        
        Omega = Omega - np.dot(B.T, np.dot(B, Omega))
        it_count += 1
        #print('Current accuracy = %1.12f | l = %d | shape(Q) = %s' % (last_errors[1],l, str(np.shape(Q))))

    B = mult(Q.conj().T, A)
           
    (R, s, Va) = svd(B, full_matrices=False)    
    U = Q.dot(R)

    # Truncate the SVD.
    # Retain only the leftmost k columns of U, the uppermost k rows of Va, and the first k entries of s.
    if truncate==0: # fixed precision for stage A
        k = 0
        for singular_value in s:
            if singular_value / s[0] < eps:
                break
            k += 1

        U = U[:,0:k]
        Va = Va[0:k,:]
        s = s[0:k]
            
    elif truncate > 0: # fixed rank for stage B
        k = min(truncate, np.shape(U)[1])
        U = U[:,0:k]
        Va = Va[0:k,:]
        s = s[0:k]

    if T_flag:
        U = Va.T
        Va = U.T
        if not deepcopy_made:
            A = A.T
        
    return U, s, Va

"""
The following is partly taken from https://github.com/NelleV/COPRE/blob/master/src/stage_A.py
"""

def SRCT1(n,l):
    """
    Computes a Subsampled Random discrete cosine transform (type I)
    Ref. Wolfram | http://reference.wolfram.com/language/ref/FourierDCTMatrix.html
    
    n: number of rows of the matrix
    l: number of columns of the matrix
    """
    
    D = random_hadamard_diag(n)
    R = np.random.choice(range(1,n+1), size=l, replace=False)

    c = np.sqrt(2/(n-1))
    q = np.pi/(n-1)
    
    FR        = np.zeros((n,l))
    FR[0,:]   = c * 0.5 * np.ones((l,)) 
    
    for r in range(1,n-1):
        FR[r,:]  =  c * np.cos(q * r * R)
        
    FR[n-1,:] = c * np.array([(-1)**(k) for k in R])

    return D, FR

def SRCT2(n,l):
    """
    Computes a Subsampled Random discrete cosine transform (type II, the most commonly used type)
    Ref. Wolfram |  http://reference.wolfram.com/language/ref/FourierDCTMatrix.html
    
    n: number of rows of the matrix
    l: number of columns of the matrix
    """
    
    D = random_hadamard_diag(n)
    R = np.random.choice(range(1,n+1), size=l, replace=False)

    c = 1/np.sqrt(n)
    q = np.pi/n
    
    FR = np.zeros((n,l))
    for r in range(0,n):
        FR[r,:]  = c * np.cos(q * (r+0.5) * R)
        
    return D, FR

def SRCT3(n,l):
    """
    Computes a Subsampled Random discrete cosine transform (type III, often referred to as the "inverse DCT")
    Ref. Wolfram |  http://reference.wolfram.com/language/ref/FourierDCTMatrix.html
    
    n: number of rows of the matrix
    l: number of columns of the matrix
    """
    
    D = random_hadamard_diag(n)
    R = np.random.choice(range(1,n+1), size=l, replace=False)

    c = 1/np.sqrt(n)
    q = np.pi/n
    
    FR = np.zeros((n,l))
    FR[0,:]   = c * np.ones((l,))
    
    for r in range(1,n):
        FR[r,:]  = c * 2 * np.cos(q * r * (R+0.5))
        
    return D, FR

def SRCT4(n,l):
    """
    Computes a Subsampled Random discrete cosine transform (type IV)
    Ref. Wolfram |  http://reference.wolfram.com/language/ref/FourierDCTMatrix.html
    
    n: number of rows of the matrix
    l: number of columns of the matrix
    """
    
    D = random_hadamard_diag(n)
    R = np.random.choice(range(1,n+1), size=l, replace=False)

    c = 2/np.sqrt(n)
    q = np.pi/n
    
    FR = np.zeros((n,l))
    for r in range(0,n):
        FR[r,:]  = c * np.cos(q * (r+0.5) * (R+0.5))
        
    return D, FR

def SRHT(n,l):
    """
    Computes a Subsampled Random discrete Hartley transform

    n: number of rows of the matrix
    l: number of columns of the matrix
    """
    D = random_hadamard_diag(n)
    R = np.random.choice(range(1,n+1), size=l, replace=False)

    FR = (2*np.pi / n) * np.outer(range(1,n+1),R) #np.array([k*R for k in range(1,n+1)])
    FR = np.cos(FR) + np.sin(FR)
    
    return D, FR

def SRHadT(n,l):
    """
    Computes a subsampled random discrete Hadamard transform

    n: number of rows of the matrix
    l: number of columns of the matrix
    """
    
    #assert np.log2(n).is_integer() 
    
    D = random_hadamard_diag(n)
    F = hadamard(n)
    R = np.random.choice(range(0,n), size=l, replace=False)

    return D, F, R

def mult_Had(A, l):
    '''
    Computes the product A P, where P is the Hadamard sampling matrix.
    '''
    m,n = np.shape(A)
    
    if np.log2(n).is_integer():           # Hadamard matrix only exists for powers of 2
        T = np.random.choice(range(0,n), size=l, replace=False)
        D = random_hadamard_diag(n)
        
        res = rec_hadamard(np.log2(n), D * A.T).T
        return res[:,T]

    else:
        next_power_of_two = int(2**(np.ceil(np.log2(n))))        # bitwise methods can work faster, but this is not the expensive step
        # Need to pad A with zeros
        A_padded = np.hstack((A, np.zeros((m,next_power_of_two-n))))
        
        T = np.random.choice(range(0,next_power_of_two), size=l, replace=False)
        D = random_hadamard_diag(next_power_of_two)

        res = rec_hadamard(np.log2(next_power_of_two), D * A_padded.T).T
        return res[:,T]
    
def rec_hadamard(k,A):
    if k == 0:
        return A
    
    A_u = A[0:int(2**(k-1)),:]
    A_l = A[int(2**(k-1)):,:]

    m_u = rec_hadamard(k-1,A_u)
    m_l = rec_hadamard(k-1,A_l)
    
    m = np.vstack((m_u+m_l, m_u-m_l))
    
    return m

def SRFT(n, l):
    """
    Computes a subsampled random Fourier Transform matrix factorization

    n: number of rows of the matrix
    l: number of columns of the matrix
    """
    
    D = random_complex_diag(n)
    #D = random_hadamard_diag(n)
    
    F = UDFT(n)
    R = np.random.choice(range(0,n), size=l, replace=False)

    return D, F, R


def UDFT(n):
    """
    Computes a unitary discrete transform (DFT) matrix.

    n: size of the n x n matrix to compute
    """

    p = np.arange(0, n).repeat(n).reshape((n, n))
    F = np.sqrt(n) * np.exp(-2 * 1j * np.pi * p * p.T / n)
    return F

def random_hadamard_diag(n):
    """
    Diagonal matrix with entries +1 or -1, randomly chosen.
    
    n: size of the matrix to compute
    """
    diag_entries = np.random.choice(np.array([-1,1]), n)
    return spdiags(diag_entries, 0, n, n)

def random_complex_diag(n):
    """
    Diagonal matrix with entries randomly distributed on the complex united circle

    n: size of the matrix to compute
    """
    a = np.random.normal(size=(n,)) + 1j * np.random.normal(size=(n,))
    return spdiags(a / np.linalg.norm(a), 0, n, n)


def mult(A, B):
    """
    default matrix multiplication.

    Multiplies A and B together via the "dot" method.

    Parameters
    ----------
    A : array_like
        first matrix in the product A*B being calculated
    B : array_like
        second matrix in the product A*B being calculated

    Returns
    -------
    array_like
        product of the inputs A and B

    Examples
    --------
    >>> from fbpca import mult
    >>> from numpy import array
    >>> from numpy.linalg import norm
    >>>
    >>> A = array([[1., 2.], [3., 4.]])
    >>> B = array([[5., 6.], [7., 8.]])
    >>> norm(mult(A, B) - A.dot(B))

    This example multiplies two matrices two ways -- once with mult,
    and once with the usual "dot" method -- and then calculates the
    (Frobenius) norm of the difference (which should be near 0).
    """

    if issparse(B) and not issparse(A):
        # dense.dot(sparse) is not available in scipy.
        return B.T.dot(A.T).T
    else:
        return A.dot(B)


def set_matrix_mult(newmult):
    """
    re-definition of the matrix multiplication function "mult".

    Sets the matrix multiplication function "mult" used in fbpca to be
    the input "newmult" -- which must return the product A*B of its two
    inputs A and B, i.e., newmult(A, B) must be the product of A and B.

    Parameters
    ----------
    newmult : callable
        matrix multiplication replacing mult in fbpca; newmult must
        return the product of its two array_like inputs

    Returns
    -------
    None

    Examples
    --------
    >>> from fbpca import set_matrix_mult
    >>>
    >>> def newmult(A, B):
    ...     return A*B
    ...
    >>> set_matrix_mult(newmult)

    This example redefines the matrix multiplication used in fbpca to
    be the entrywise product.
    """

    global mult
    mult = newmult

