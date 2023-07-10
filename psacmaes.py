# coding:utf-8

import sys
import os
from abc import ABCMeta, abstractmethod, abstractproperty
import time
import pprint
import math
from collections import deque
from functools import partial

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.core.umath import floor
from numpy.random.mtrand import rand

try:
    import quadprog
except ImportError:
    print("`quadprog` not found. Try `pip install quadprog`.")
    print("Some constraint handling techniques are unavailable.")

mpl.rc('lines', linewidth=2, markersize=8)
mpl.rc('font', size=12)
mpl.rc('grid', color='0.75', linestyle=':')
mpl.rc('ps', useafm=True)  # Force to use
mpl.rc('pdf', use14corefonts=True)  # only Type 1 fonts
mpl.rc('text', usetex=True)  # for a paper submision

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

NITER = 0
MEASURE = 1
ELAPSED = 2
CRITERION = 3
PROFILE = 'profile'
EXT = '.dat'
DELIM = '_'
PATTERNS = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*", ""]

if sys.version_info[0] >= 3:
    # For Python Version >= 3.0
    basestring = str
else:
    # For Python Version < 3.0    
    range = xrange


class CmaSolution:
    """Cma Solution Class

    Attributes
    ----------
    arxraw, arxmut, arxfeas : numpy.ndarray[float]
        raw solution, mutation vector, repaired solution
        They are all 1 dimensional, and must have the same shape
    arffeas, arpen, arfit : float
        objective value at repaired solution, penalty value, (final) fitness value
        They are all 1 dimensional.
        """
    def __init__(self, array_like):
        """
        Parameters
        ----------
        array_like : array_like
            solution vector to be set to `arxfeas`
        """
        self.arxfeas = np.asarray(array_like)
        self.arxraw = np.copy(self.arxfeas)
        self.arxmut = np.zeros(len(self.arxfeas))
        self.x_to_eval = np.copy(self.arxfeas)
        self.arffeas = np.nan
        self.arpen = np.nan
        self.arfit = np.nan


class NormalOrderStatistics(object):
    """Compute Moments of Normal Order Statistics

    Requires
    --------
    numpy
    scipy.stats.norm
    """

    def __init__(self, n):
        """Normal Order Statistics from `n` populations

        Normal order statistics of population size `n` are the ordered 
        random variables
            N_{1:n} < N_{2:n} < ... < N_{n:n}
        that are drawn from the standard normal distribution N(0, 1)
        independently.

        Parameters
        ----------
        n : int
            population size
        """
        self._n = n
        self._pr = np.arange(1, n + 1, dtype=float) / (n + 1)
        self._q0r = norm.ppf(self._pr)
        self._q1r = 1.0 / norm.pdf(self._q0r)
        self._q2r = self._q0r * self._q1r**2
        self._q3r = (1.0 + 2.0 * self._q0r**2) * self._q1r**3
        self._q4r = self._q0r * (7.0 + 6.0 * self._q0r**2) * self._q1r**4

    def exp(self):
        """Expectation of the normal order statistics, using Taylor Expansion.

        Returns
        -------
        1D ndarray : array of expectation of the normal order statistics

        Algorithm
        ---------
        Eq. (4.6.3)--(4.6.5) combined with Example 4.6 in "Order Statistics".
        """
        result = self._q0r
        result += self._pr * (1 - self._pr) * self._q2r / (2 * self._n + 4)
        result += self._pr * (1 - self._pr) * (
            1 - 2 * self._pr) * self._q3r / (3 * (self._n + 2)**2)
        result += (self._pr *
                   (1 - self._pr))**2 * self._q4r / (8 * (self._n + 2)**2)
        return result

    def var(self):
        """Variance of the normal order statistics, using Taylor Expansion.

        Returns
        -------
        1D ndarray : array of variance of the normal order statistics

        Algorithm
        ---------
        Eq. (4.6.3)--(4.6.5) combined with Example 4.6 in "Order Statistics".
        """
        result = self._pr * (1 - self._pr) * self._q1r**2 / (self._n + 2)
        result += self._pr * (1 - self._pr) * (
            1 - 2 * self._pr) * 2 * self._q1r * self._q2r / ((self._n + 2)**2)
        result += (self._pr * (1 - self._pr))**2 * (
            self._q1r * self._q3r + self._q2r**2 / 2) / (self._n + 2)**2
        return result

    def cov(self):
        """Covariance of the normal order statistics, using Taylor Expansion.

        Returns
        -------
        2D ndarray : array of covariance of the normal order statistics

        Algorithm
        ---------
        Eq. (4.6.3)--(4.6.5) combined with Example 4.6 in "Order Statistics".
        """
        result = np.outer(self._pr**2 * self._q2r, (1 - self._pr)
                          **2 * self._q2r) / 2
        result += np.outer(self._pr * (1 - 2 * self._pr) * self._q2r,
                           (1 - self._pr) * self._q1r)
        result += np.outer(self._pr * self._q1r,
                           (1 - self._pr) * (1 - 2 * self._pr) * self._q2r)
        result += np.outer(self._pr**2 * (1 - self._pr) * self._q3r,
                           (1 - self._pr) * self._q1r) / 2
        result += np.outer(self._pr * self._q1r, self._pr * (1 - self._pr)
                           **2 * self._q3r) / 2
        result /= (self._n + 2)**2
        result += np.outer(self._pr * self._q1r,
                           (1 - self._pr) * self._q1r) / (self._n + 2)
        return np.triu(result) + np.triu(result, k=1).T

    def blom(self):
        """Blom's Approximation of the Expectation of Normal Order Statistics

        Returns
        -------
        1D ndarray : array of expectation of the normal order statistics
        """
        alpha = 0.375
        pir = (np.arange(1, self._n + 1) - alpha) / (self._n + 1 - 2 * alpha)
        return norm.ppf(pir)

    def davis_stephens(self):
        """Refinement of Covariance Matrix by Algorithm 128

        Returns
        -------
        2D ndarray : array of covariance of the normal order statistics

        See
        ---
        https://statistics.stanford.edu/sites/default/files/SOL%20ONR%20254.pdf
        """
        result = self.cov()
        n = self._n
        for i in range((n + 1) // 2):
            rowsum = np.sum(result[i])
            free = np.sum(result[i, i:n - i])
            result[i, i:n - i] *= 1 + (1 - rowsum) / free
            result[i:n - i, i] = result[i, i:n - i]
            result[n - i - 1, i:n - i] = (result[i, i:n - i])[::-1]
            result[i:n - i, n - i - 1] = (result[i, i:n - i])[::-1]
        return result


def sigma_normalization_factor(dim, weights, cm=1.):
    """sigma = snf * ||xmean|| """
    nos = NormalOrderStatistics(len(weights))
    nlam = nos.blom()
    beta = -np.dot(nlam, weights)
    if len(weights) < 50:
        nnlam = nos.davis_stephens()
        gamma = beta**2 + np.dot(np.dot(nnlam, weights), weights)
    else:
        gamma = beta**2
    muw = np.sum(np.abs(weights)) ** 2 / np.dot(weights, weights)
    return beta * muw / (dim - 1 + gamma * muw) / cm


def compute_nearest_feasible(vec_x, mat_inv_c, mat_a, vec_b):
    """Find the nearest feasible solution

    It solve the following convex quadratic programming:

    argmin_{y in R^N} f(y) = (x - y) * C^{-1} * (x - y) / N
    subject to A*y <= b

    Parameters
    ----------
    vec_x : 1d array-like
        x
    mat_inv_c : 2d array-like
        C^{-1}
    mat_a : 2d array-like
        A
    vec_b : 1d array-like
        b

    Returns
    -------
    y (1d array-like) : solution
    f (float) : f(y)
    iact (1d array-like) : list of indeces of the active constraints at y

    Note
    ----
    The solver uses quadprog package.
    The following is taken from the solve_qp docstring.

    ============================================================================
    Solve a strictly convex quadratic program
        Minimize     1/2 x^T G x - a^T x
        Subject to   C.T x >= b
        This routine uses the the Goldfarb/Idnani dual algorithm [1].
        References
        ---------
        ... [1] D. Goldfarb and A. Idnani (1983). A numerically stable dual
            method for solving strictly convex quadratic programs.
            Mathematical Programming, 27, 1-33.
        Parameters
        ----------
        G : array, shape=(n, n)
            matrix appearing in the quadratic function to be minimized
        a : array, shape=(n,)
            vector appearing in the quadratic function to be minimized
        C : array, shape=(n, m)
            matrix defining the constraints under which we want to minimize the
            quadratic function
        b : array, shape=(m), default=None
            vector defining the constraints
        meq : int, default=0
            the first meq constraints are treated as equality constraints,
            all further as inequality constraints (defaults to 0).
        factorized : bool, default=False
            If True, then we are passing :math:`R^{−1}` (where :math:`G = R^T R`)
            instead of the matrix G in the argument G.
        Returns
        -------
        x : array, shape=(n,)
            vector containing the solution of the quadratic programming problem.
        f : float
            the value of the quadratic function at the solution.
        xu : array, shape=(n,)
            vector containing the unconstrained minimizer of the quadratic function
        iterations : tuple
            2-tuple. the first component contains the number of iterations the
            algorithm needed, the second indicates how often constraints became
            inactive after becoming active first.
        lagrangian : array, shape=(m,)
            vector with the Lagragian at the solution.
        iact : array
            vector with the indices of the active constraints at the solution.
    ============================================================================

    In our case,
    G = C^{-1}
    a^T = C^{-1} * x
    -C.T = A
    b = -b
    """
    inv_c_x = np.dot(mat_inv_c, vec_x)
    scaling1 = np.max(np.abs(mat_inv_c))
    scaling2 = np.max(np.abs(mat_a))
    y, f, _, _, _, iact = quadprog.solve_qp(G=mat_inv_c / scaling1,
                                            a=inv_c_x / scaling1,
                                            C=-mat_a.T / scaling2,
                                            b=-vec_b / scaling2,
                                            meq=0,
                                            factorized=False)
    f = ((2.0 * scaling1) * f + np.dot(vec_x, inv_c_x)) / len(vec_x)
    return y, f, iact



def ranking(v):
    """Index sort, Lower rank, Upper rank

    Examples
    --------
    v = np.array([1, 1, 0, 3])
    idx, lower, upper = rank(v)

    idx   == [2, 0, 1, 3] or [2, 1, 0, 3]
    lower == [0, 1, 1, 3]
    upper == [0, 2, 2, 3]

    Parameters
    ----------
    v : array-like

    Returns
    -------
    idx : idx[j] is the index of `j`th smallest among `v`
    lower : lower[j] is the lower rank of the index idx[j]
    upper : upper[j] is the upper rank of the index idx[j]
    """
    idx = np.argsort(v)
    lower = np.arange(v.shape[0], dtype=int)
    upper = np.arange(v.shape[0], dtype=int)
    dup = np.where(np.abs(v[idx[1:]] - v[idx[:-1]]) <= np.spacing(v[idx[1:]]))[0]
    if dup.shape[0] > 0:
        jump = np.where(dup[1:] - dup[:-1] > 1)[0]
        if jump.shape[0] == 0:
            lower[dup + 1] = lower[dup[0]]
            upper[dup] = upper[dup[-1] + 1]
        else:
            j = 0
            for i in jump:
                lower[dup[j:i+1] + 1] = lower[dup[j]]
                upper[dup[j:i+1]] = upper[dup[i] + 1]
                j = i+1
            lower[dup[j:] + 1] = lower[dup[j]]
            upper[dup[j:]] = upper[dup[-1] + 1]
    return idx, lower, upper


def resample(dim, sample_func, func_list, non_positive_indices=None, maxresample=100, cls=CmaSolution):
    """Resampling Infeasibility Handling

    The functions will
    * raise Exception or
    * return np.inf or np.nan or None
    if they are not well-defined

    It resamples a candidate solution until its function values are well-defined.
    Moreover, if non_positive_indeces is given, it resamples until the corresponding
    function values will be non positive.

    Parameters
    ----------
    dim : int
        dimension
    sample_func : callable
        function to generate a candidate solution.
        It takes a list of CmaSolusion instance as an argument
    func_list : list of callable
        list of functions for which the candidate solutions are resampled
        until they are feasible
    non_positive_indices : list of indeces, optional
        candidate solutions are resampled until the functions with the given indices
        return non-positive values
    maxresample : int, optional
        maximum number of resample
    cls : class, optional
        Solusion Class, subclass of CmaSolution
    """
    assert issubclass(cls, CmaSolution)
    non_positive_indices = non_positive_indices if non_positive_indices is not None else []
    neval_func = [0] * len(func_list)
    idx_feas = maxresample
    resampled = []
    for i in range(maxresample):
        indi = cls(np.zeros(dim))
        is_feas = True
        indi = sample_func([indi])
        indi[0].values = [None] * len(func_list)
        for j, f in enumerate(func_list):
            try:
                indi[0].values[j] = f(indi[0])
            except:
                indi[0].values[j] = np.inf
            if indi[0].values[j] in (None, np.nan):
                indi[0].values[j] = np.inf
            neval_func[j] += 1
            if (indi[0].values[j] == np.inf) or ((j in non_positive_indices) and (indi[0].values[j] > 0)):
                is_feas = False
                break
        resampled += indi
        if is_feas:
            idx_feas = i
            break
    return idx_feas, resampled, neval_func


class ABCIterativeAlgorithm(object):
    """Iterative Algorithm Interface
    Main functionalities are: onestep, check, log, run, plotme, and plot
    onestep : perform one iteration of the algorithm, 
              call _onestep
    check   : check if a termination condition is satisfied, 
              call _check
    log     : write and display the internal states, 
              call _log_preprocess before taking a log
    run     : perform onestep-check-log loop untile 
              a termination condition is satisfied
    plotme  : plot the log data, shortcut of `plot`
    plot    : plot the log data
    Derived Classes need to implement the following methods and properties
    
    _onestep : one iteration of the algorithm
    _check   : termination checker 
    _log_preprocess : optional, preprocessing for log output.
                      mainly for monkeypatch.
    measure   : measure (int) of the run-length of the algorithm
    criterion : measure (float) of the progress of the algorithm
    recommendation : current recommendation (ndarray)
    See
    ---
    RandomSearch, RandomSearchOption
    """
    __metaclass__ = ABCMeta

    def __init__(self, opts):
        """
        Parameters
        ----------
        opts : IterativeAlgorithmOption or its derived class instance
            It can be either parsed or not parsed.
        """
        self._niter = 0
        self._time_step = 0.0
        self._time_check = 0.0
        self._time_log = 0.0
        self._is_satisfied = False
        self._condition = ''
        assert isinstance(opts, IterativeAlgorithmOption)
        if opts.is_parsed:
            self.opts = opts
        else:
            self.opts = opts.parse()
            self.opts_original = opts
        # Logger dictionary
        self.logger = dict()
        if self.opts.log_prefix and self.opts.log_span > 0:
            self.profile_logger = self.opts.log_prefix + DELIM + PROFILE + EXT
            with open(self.profile_logger, 'w') as f:
                f.write('#' + type(self).__name__ + "\n")
        for key in self.opts.log_variable_list:
            self.logger[key] = self.opts.log_prefix + DELIM + key + EXT
            with open(self.logger[key], 'w') as f:
                f.write('#' + type(self).__name__ + "\n")
    
    @property
    def is_satisfied(self):
        return self._is_satisfied
    
    @property
    def condition(self):
        return self._condition

    def onestep(self):
        t0 = time.perf_counter()
        self._onestep()
        self._time_step += time.perf_counter() - t0
        self._niter += 1

    def check(self):
        t0 = time.perf_counter()
        self._is_satisfied, self._condition = self._check()
        self._time_check += time.perf_counter() - t0
        return self._is_satisfied, self._condition

    def log(self, flgforce=False, condition=''):
        """Take a log
        Parameters
        ----------
        flgforce : bool, optional
            force to take a log.
        condition : str, optional
            termination condition.
        """

        elapsed_time = self.time_step + self.time_check
        # Check if the log should be taken
        if ((flgforce and self.opts.log_span > 0) or
            (self.opts.log_span >= 1 and self.niter % self.opts.log_span == 0)
                or (1.0 > self.opts.log_span > 0.0 and
                    elapsed_time * self.opts.log_span > self.time_log)):

            t0 = time.perf_counter()
            # Preprocess
            self._log_preprocess()
            # Display
            displayline = "{0} {1} {2} {3}".format(
                repr(self.niter),
                repr(self.measure), repr(elapsed_time), str(self.criterion))
            if self.opts.log_display:
                print(displayline)
                if condition:
                    print('# End with condition = ' + condition)
            with open(self.profile_logger, 'a') as f:
                f.write(displayline + "\n")
                if condition:
                    f.write('# End with condition = ' + condition)
            for key, log in self.logger.items():
                key_split = key.split('.')
                key = key_split.pop(0)
                var = getattr(self, key)  # var = self.__dict__[key] # Y.A. on 2017/09/22
                for i in key_split:
                    var = getattr(var, i)  # var.__dict__[i] # Y.A. on 2017/09/22
                if isinstance(var, np.ndarray) and len(var.shape) > 1:
                    var = var.flatten()
                varlist = np.hstack(
                    (self.niter, self.measure, elapsed_time, var))
                with open(log, 'a') as f:
                    f.write(' '.join(map(repr, varlist)) + "\n")
            self._time_log += time.perf_counter() - t0

    def _log_preprocess(self):
        """Preprocess for logging
        This function is called at the top of `log` function.
        By default, it does nothing. This method will be implemented
        when one wants to record some values that are not the attribute
        of the `iterative_algorithm`.
        Two possible usage of this functions are:
            1. to implement this method in a derived class
            2. to monkeypatch this method outside the class definition
        An example of the second usage is as follows
        Example
        -------
        Assume that we want to record the product of `a` and `b`
        defined in the `iterative_algorithm`.
        >>> # Define a function to be monkeypatched
        >>> def monkeypatch(self):
        >>>    a = self.iterative_algorithm.a
        >>>    b = self.iterative_algorithm.b
        >>>    # set a variable to `self.iterative_algorithm`
        >>>    self.iterative_algorithm.a_times_b = a * b
        >>> # Replace `_log_preprocess`
        >>> ABCIterativeAlgorithm._log_preprocess = monkeypatch
        """
        pass

    def run(self):
        """Run the iterative algorithm
        If you want to monitor the behavior of the algorithm during the run,
        it is possible to do it by either using the multithreading or run a different interpreter
        Example 1
        ---------
        from threading import Thread
        th = Thread(target=algo.run, name='algo.run') # algo is an instance of this class
        th.start()
        algo.plotme()
        Example 2
        ---------
        algo.run()
        # Open a separate interpreter
        ABCIterativeAlgorithmOption.plot(...)
        """
        is_satisfied = False
        while not is_satisfied:
            if self.opts.check_exception:
                try:
                    self.onestep()
                    is_satisfied, condition = self.check()
                    self.log(flgforce=is_satisfied, condition=condition)
                except Exception as e:
                    is_satisfied = True
                    self.log(flgforce=True, condition='exception: ' + str(e))
            else:
                self.onestep()
                is_satisfied, condition = self.check()
                self.log(flgforce=is_satisfied, condition=condition)

    def plotme(self, xaxis=MEASURE, **kwargs):
        """Plot the results using `plot` function
        Parameters
        ----------
        xaxis : int
            see xaxis in `plot`
        kwargs : optional parameters
            optional arguments to `plot`
        """
        return ABCIterativeAlgorithm.plot(opts=self.opts, xaxis=xaxis, **kwargs)

    @staticmethod
    def plot(prefix='',
             xaxis=MEASURE,
             variable_list=None,
             opts=None,
             ncols=None,
             figsize=None,
             cmap_='winter'):
        """Plot the result
        This allows to plot previously generated or post-processed data that
        has the same format as the log file generated by ``log``.
        Parameters
        ----------
        prefix : str
            the prefix of the log file
        xaxis : int
            NITER == 0. vs iter
            MEASURE == 1. vs measure
            ELAPSED == 2. vs cpu time in sec.
        variable_list : list of str
            names of variables
        opts : OptionBaseClass or its derived class
            If this is given, the other parameters are not needed
        Returns
        -------
        fig : figure object.
            figure object
        axdict : dictionary of axes
            the keys are the names of variables given in `variable_list`
        The log files must be located at ``prefix`` + PROFILE + EXT,
        and ``prefix`` + ``variable_list[i]`` + EXT.
        """
        if opts:
            prefix = opts.log_prefix
            variable_list = opts.log_variable_list
        if variable_list is None:
            variable_list = []

        # Default settings
        nfigs = 1 + len(variable_list)
        if ncols is None:
            ncols = int(np.ceil(np.sqrt(nfigs)))
        nrows = int(np.ceil(nfigs / ncols))
        if figsize is None:
            figsize = (4 * ncols, 3 * nrows)
        axdict = dict()
        # Figure
        fig = plt.figure(figsize=figsize)
        # The first figure
        x = np.loadtxt(prefix + DELIM + PROFILE + EXT)
        x = x[~np.isnan(x[:, xaxis]), :]  # remove columns where xaxis is nan
        # Axis
        ax = plt.subplot(nrows, ncols, 1)
        ax.set_title('criterion')
        ax.grid(True)
        ax.grid(which='major', linewidth=0.50)
        ax.grid(which='minor', linewidth=0.25)
        plt.plot(x[:, xaxis], x[:, 3:])
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
        axdict['criterion'] = ax

        # The other figures
        idx = 1
        for key in variable_list:
            idx += 1
            x = np.loadtxt(prefix + DELIM + key + EXT)
            x = x[~np.isnan(
                x[:, xaxis]), :]  # remove columns where xaxis is nan
            ax = plt.subplot(nrows, ncols, idx)
            ax.set_title(r'\detokenize{' + key + '}')
            ax.grid(True)
            ax.grid(which='major', linewidth=0.50)
            ax.grid(which='minor', linewidth=0.25)
            if False:  # Before 2017/02/11
                plt.plot(x[:, xaxis], x[:, 3:])
            else:  # New
                cmap = plt.get_cmap(cmap_)
                cNorm = mpl.colors.Normalize(vmin=0, vmax=x.shape[1] - 1)
                scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
                for i in range(x.shape[1] - 3):
                    plt.plot(
                        x[:, xaxis], x[:, 3 + i], color=scalarMap.to_rgba(i))
            ax.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(my_formatter))
            ax.yaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(my_formatter))
            axdict[key] = ax

        plt.tight_layout() # NOTE: not sure if it works fine
        return fig, axdict

    @property
    def niter(self):
        return self._niter

    @property
    def time_step(self):
        return self._time_step

    @property
    def time_check(self):
        return self._time_check

    @property
    def time_log(self):
        return self._time_log

    @abstractmethod
    def _onestep(self):
        """Do one step (one iteration) of the algorithm"""
        pass

    @abstractmethod
    def _check(self):
        """Check the termination criteria
        Returns
        -------
        flag : bool
            True if one of the termination condition is satisfied, False otherwise
        condition : str
            String of condition, empty string '' if no condition is satisfied
        """
        if self.criterion < self.opts.check_target:
            return True, 'target'
        elif self.niter >= self.opts.check_maxiter:
            return True, 'maxiter'
        elif self.time_step + self.time_check >= self.opts.check_maxsec:
            return True, 'maxsec'
        elif self.measure >= self.opts.check_maxruntime:
            return True, 'maxruntime'
        else:
            return False, ''

    @abstractproperty
    def measure(self):
        """Measure (int) of the run-length of the algorithm"""

    @abstractproperty
    def criterion(self):
        """Measure (float) of the progress of the algorithm"""

    @abstractproperty
    def recommendation(self):
        """The current recommendation (ndarray)"""


def my_formatter(x, pos):
    """Float Number Format for Axes"""
    float_str = "{0:2.1e}".format(x)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0}e{1}".format(base, int(exponent))
    else:
        return r"" + float_str + ""


def _myeval(s, g, l):
    if isinstance(s, basestring):
        return eval(s, g, l)
    else:
        return s


class OptionBaseClass(object):
    def __init__(self, **kwargs):
        """OptionBaseClass
        
        It is a container of variables, whose values can depends on the other variables.
        The method `parse` automatically determines the right order to parse the variables
        and returns a new instance consisting of the parsed values.
        Parameters
        ----------
        The pair (key, value) in the keyward arguments is stored in the instance as 
            
            instance.key = value
        """
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.is_parsed = False

    def __str__(self):
        return type(self).__name__ + str(self.__dict__)

    def setattr_from_local(self, locals_):
        for key in locals_:
            if key != 'self' and key != 'kwargs':
                setattr(self, key, locals_[key])

    def disp(self):
        """Display all the options"""
        print(type(self).__name__)
        pp = pprint.PrettyPrinter()
        pp.pprint(self.__dict__)

    def to_latex(self):
        """Parse the option to LaTeX
        Return
        ------
        LaTeX string. 
        """
        res = ''
        res += '\\begin{longtable}{ll}\n'
        res += '\\caption{The parameters of ' + type(self).__name__ + '.}\\\\\n'
        res += '\\hline\n'
        res += 'Key & Value \\\\\n'
        res += '\\hline\n'
        for key in sorted(list(self.__dict__)):
            if key != 'is_parsed':
                res += '\\detokenize{' + str(key) + '} & \\detokenize{' + str(getattr(self, key)) + '}\\\\\n'
        res += '\\hline\n'
        res += '\\end{longtable}\n'
        return res

    def parse(self, env=None, flg_print=False):
        """Parse the member variables that are string expressions
        Parameters
        ----------
        env : dict, default None
            The dictionary of a namespace. A typical choice is globals().
            The values in the instance are evaluated on the environment `env`
             updated with globals() called inside the method.
        flg_print : bool, default False
            Print the parse process.
        Returns
        -------
        parsed : an instance of the same class as `self`
            all the member variables are parsed.
        Example
        -------
        Case 1.
        >>> opts = OptionBaseClass(N='10', M='int(N)', L='int(N)')
        >>> parsed = opts.parse()
        >>> parsed.disp()
        OptionBaseClass
        {'L': 10, 'M': 10, 'N': 10, 'is_parsed': True}
        Case 2.
        repr(N) is evaluated before it is passed to the function
        >>> N = 100
        >>> opts = OptionBaseClass(N='10', L=repr(N))
        >>> parsed = opts.parse()
        >>> parsed.disp()
        OptionBaseClass
        {'L': 100, 'N': 10, 'is_parsed': True}
        Case 3.
        >>> N = 100
        >>> opts = OptionBaseClass(M='N', L=repr(N))
        >>> parsed = opts.parse(globals())
        >>> parsed.disp()
        OptionBaseClass
        {'L': 100, 'M': 100, 'is_parsed': True}
        In the following cases, one may obtain unexpected outputs.
        Case A.
            opts = OptionBaseClass(N='10', M='N')
        The output of `parse` method is undefined.
        If `M` is evaluated before `N`, the result will be M = '10' (string).
        To prevent this unexpected behavior, consider to cast the variable like
            opts = OptionBaseClass(N='10', M='int(N)')
        Case B.
            opts = OptionBaseClass(N='M', M='N')
        Obviously, the variables can not be evaluated if there is a pair of
        variables that depend on each other.
        Case C.
            N = 100
            mypow = pow
            opts = OptionBaseClass(M='mypow(N, L)', L='2')
            parsed = opts.parse(globals())
        To refer to variables and objects defined in the caller,
        call `parse` with env=globals()
        Case D.
        Call `parse` with env=globals() if some modules are required
        to evaluate some variables
            import numpy as np
            opts = OptionBaseClass(N='np.arange(5)', M='np.sqrt(N)')
            parsed = opts.parse(globals())
        """
        if self.is_parsed:
            print("Already parsed. Returns itself.")
            return self

        parsed_dict = dict()
        failure_count = 0
        key_list = list(self.__dict__)
        key_list.remove('is_parsed')

        if env is None:
            env = dict(globals())
        else:
            env = dict(env)
        env.update(globals())
        env.update(self.__dict__)

        while key_list:
            if failure_count >= len(key_list):
                print("Some options couldn't be parsed: " + str(key_list))
                print("Their values are as follows.")
                for key in key_list:
                    print(key + ': ' + getattr(self, key))

                print("\n" + "To find out the reason, see the document of " +
                      "`OptionBaseClass.parse`, and\n" +
                      "A. type-cast the option variables (see Case A);\n" +
                      "B. remove a cycle of variables (see Case B);\n" +
                      "C. call `parse` with env=globals() " +
                      "to use global variables (see Case C);\n" +
                      "D. import modules and functions such as " +
                      "`numpy`, `exp`, `sqrt` (see Case D).\n")

                print("Here are the parsed values:")
                pp = pprint.PrettyPrinter()
                pp.pprint(parsed_dict)

                print("Try 'OptionBaseClass.parse' with 'flg_print' option.")
                raise ValueError()

            key = key_list.pop()
            try:
                val = _myeval(getattr(self, key), env, parsed_dict)
                parsed_dict[key] = val
                if flg_print:
                    print(key + ': ' + repr(val))
                failure_count = 0
            except:
                key_list.insert(0, key)
                failure_count += 1

        parsed = self.create()
        key_list = list(self.__dict__)
        key_list.remove('is_parsed')
        for key in key_list:
            setattr(parsed, key, parsed_dict[key])
        parsed.is_parsed = True
        return parsed

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)


class IterativeAlgorithmOption(OptionBaseClass):
    """Option Container for Iterative Algorithm Class"""
    def __init__(self,
                 log_prefix=repr('.'),
                 log_span=repr(0.1),
                 log_display=repr(True),
                 log_variable_list=repr([]),
                 check_target="-np.inf",
                 check_maxiter="np.inf",
                 check_maxsec="np.inf",
                 check_maxruntime="np.inf",
                 check_exception=repr(False),
                 **kwargs):

        OptionBaseClass.__init__(self, **kwargs)
        self.setattr_from_local(locals())

class TolfChecker(object):
    def __init__(self, size=20):
        """
        Parameters
        ----------
        size : int
            Nnumber of points for which the value is restored
            For CMA-ES, 10 + (N / lambda) will be a reasonable value.
        """
        self._min_hist = np.empty(size) * np.nan
        self._l_quartile_hist = np.empty(size) * np.nan
        self._median_hist = np.empty(size) * np.nan
        self._u_quartile_hist = np.empty(size) * np.nan
        self._max_hist = np.empty(size) * np.nan
        self._pop_hist = np.empty(size) * np.nan
        self._next_position = 0

    def update(self, arf):
        """Add values to the history and remove elements from history if needed"""
        self._min_hist[self._next_position] = np.nanmin(arf)
        self._l_quartile_hist[self._next_position] = np.nanpercentile(arf, 25)
        self._median_hist[self._next_position] = np.nanmedian(arf)
        self._u_quartile_hist[self._next_position] = np.nanpercentile(arf, 75)
        self._max_hist[self._next_position] = np.nanmax(arf)
        self._next_position = (
            self._next_position + 1) % self._min_hist.shape[0]

    def check(self, tolfun=1e-9):
        # alias to check_absolute
        return self.check_relative(tolfun)

    def check_relative(self, tolfun=1e-11):
        iqr = np.nanmedian(self._u_quartile_hist - self._l_quartile_hist)
        med_ = abs(np.nanmedian(self._min_hist))
        if not np.isnan(iqr) and not np.isnan(med_):
            return iqr < tolfun * med_
        else:
            return False

    def check_absolute(self, tolfun=1e-11):
        iqr = np.nanmedian(self._u_quartile_hist - self._l_quartile_hist)
        if not np.isnan(iqr):
            return iqr < tolfun
        else:
            return False

    def check_flatarea(self):
        return np.all(self._l_quartile_hist - self._min_hist == 0)


class TolxChecker(object):
    def __init__(self, dim, size=20):
        """
        Parameters
        ----------
        size : int
            number of points for which the value is restored
        """
        self._min_hist = np.empty((size, dim)) * np.nan
        self._l_quartile_hist = np.empty((size, dim)) * np.nan
        self._median_hist = np.empty((size, dim)) * np.nan
        self._u_quartile_hist = np.empty((size, dim)) * np.nan
        self._max_hist = np.empty((size, dim)) * np.nan
        self._next_position = 0

    def update(self, arx):
        """Add values to the history and remove elements from history if needed"""
        self._min_hist[self._next_position] = np.nanmin(arx, axis=0)
        self._l_quartile_hist[self._next_position] = np.nanpercentile(
            arx, 25, axis=0)
        self._median_hist[self._next_position] = np.nanmedian(arx, axis=0)
        self._u_quartile_hist[self._next_position] = np.nanpercentile(
            arx, 75, axis=0)
        self._max_hist[self._next_position] = np.nanmax(arx, axis=0)
        self._next_position = (
            self._next_position + 1) % self._min_hist.shape[0]

    def check(self, tolx=1e-10):
        # alias to check_absolute
        return self.check_relative(tolx)

    def check_relative(self, tolx=1e-10):
        iqr = np.nanmedian(self._u_quartile_hist - self._l_quartile_hist, axis=0)
        med_ = np.abs(self._median_hist).min(axis=0)
        if not np.all(np.isnan(iqr)) and not np.all(np.isnan(med_)):
            return np.any(iqr < tolx * med_)
        else:
            return False

    def check_absolute(self, tolx=1e-10):
        iqr = np.nanmedian(self._u_quartile_hist - self._l_quartile_hist, axis=0)
        if not np.all(np.isnan(iqr)):
            return np.any(iqr < tolx)
        else:
            return False


def weight(pop, xmean, transform_inverse, base_weight=None, cone=0, cmu=1):
    """Compute Weights

    Parameters
    ----------
    pop : list of CmaSolution instances
        population to be weighted
    xmean : ndarray 1d
        mean vector
    transform_inverse : callable
        function to inverse-transform a solution
    pre_weight : array-like
        pre-defined weight values
    cone, cmu : float, optional
        learning rate for rank-one and rank-mu
        0 <= cone + cmu <= 1
        If cmu = 1, the resulting weights are all non-negative

    Returns
    -------
    ndarray (1D)
        scaled weights
    """
    lam = len(pop)
    N = len(xmean)
    ww = pre_weight(N, lam=lam, base_weight=base_weight, cone=cone, cmu=cmu)
    # Treat Tie
    idx, lr, ur = ranking(np.asarray([indi.arfit for indi in pop]))
    w = np.array([np.mean(ww[lr[i]:ur[i]+1]) for i in range(lam)])
    # assert np.all(idx == pop.idx)
    # Scale Weights
    for i in range(lam):
        if w[i] < 0:
            arz = transform_inverse(pop[idx[i]].arxraw - xmean)
            w[i] *= N / np.sum(arz**2)
        #arz = transform_inverse(pop.arxraw - xmean)
        #w[w < 0] *= N / np.sum(arz[idx[w < 0]]**2, axis=1)
    return w


def pre_weight(N, lam=None, base_weight=None, cone=0, cmu=1):
    """Compute Weights

    Parameters
    ----------
    N : int
        dimension of search space
    lam : int
        pop size
    base_weight : array-like
        pre-defined weight values
    cone, cmu : float, optional
        learning rate for rank-one and rank-mu
        0 <= cone + cmu <= 1
        If cmu = 1, the resulting weights are all non-negative

    Returns
    -------
    ndarray (1D)
        pre-weights
    """
    if lam is None:
        assert base_weight is not None
        lam = len(base_weight)
    if base_weight is None:
        ww = np.array([math.log((lam + 1) / 2.0) - math.log(1 + i) for i in range(lam)])
    else:
        assert len(base_weight) == lam
        ww = np.array(base_weight, copy=True)
    # Taken from Niko's CMA-ES Tutorial (Appendix A)
    ww[ww > 0] /= np.sum(ww[ww > 0]) if np.any(ww > 0) else 1
    mueff_pos = 1.0 / np.sum(ww[ww > 0] ** 2) if np.any(ww > 0) else 0
    if cmu > 0:
        # Taken from Niko's CMA-ES Tutorial (Appendix A)
        ww[ww < 0] /= -np.sum(ww[ww < 0]) if np.any(ww < 0) else 1
        mueff_neg = 1.0 / np.sum(ww[ww < 0] ** 2) if np.any(ww < 0) else 0
        teig = max(1, 1 / (10 * N * (cmu + cone))) # eigh performed every teig iterations
        amu = (1 + cone / cmu) / (1 + N * ((20 * N / (20 * N - 1)) ** teig - 1))
        amuw = 1 +  2 * mueff_neg / (mueff_pos + 2)
        amupos = (1 - cone - cmu) / (N * cmu)
        ww[ww < 0] *= np.nanmin((amu, amuw, amupos))
    return ww


class MixedIntegerHandler(object):

    def __init__(self, stair_width, cs=None, ds=None, minstd=0, correction_mode='sigma', flg_integer_mutation=True):
        """Helper for Mixed Integer Problems

        Three functionarities are implimented:
            integer_mutation : mutation for integer variables
            mixed_integer_csa : csa variant for mixed integer
            repair_coordinate_std : force the minimum std for integer variables

        Parameters
        ----------
        stair_width : array-like
            stair width for integer variables. 0 for float variables.
        cs, ds : float
            cumulation factor and damping factor
        minstd : float or array-like, default = 0
            minimum (coordinate-wize) standard deviation
        correction_mode : str, optional, default = 'sigma'
            mode selection for covariance correction
            'none' : no correction
            'sigma' : sigma correction
            'coordinate_std' : coordinate_std correction
        flg_integer_mutation : bool, optional, default = True
            flag for integer mutation
        """
        N = len(stair_width)
        self.stair_width = stair_width
        # for update
        self.ps = np.zeros(N)
        self.cs = cs
        self.ds = ds
        self.chiN = math.sqrt(N) * (1.0 - 1.0 / (4.0 * N) + 1.0 / (21.0 * N * N))
        self.hsig = True
        self.last_sig_change_fac = 1.0
        self.correction_mode = correction_mode
        self.flg_integer_mutation = flg_integer_mutation
        # for repair_coordinate_std
        ien = np.sum(self.stair_width > 0.0)
        if np.isscalar(minstd):
            self.minstd = np.asarray([max(minstd, self.stair_width[i] / (2.0 * ien + 1.0)) for i in range(N)])
        else:
            self.minstd = np.asarray([max(minstd[i], self.stair_width[i] / (2.0 * ien + 1.0)) for i in range(N)])

    def integer_mutation(self, lam, xmean, coordinate_std, bestx=None):
        """Integer Mutation
    
        Parameters
        ----------
        lam : int
            number of candidate solutions
        xmean, coordinate_std : 1d array-like
            mean vector and the coordinate wise standard deviation (sqrt of diag of sigma**2 * C)
        bestx : 1d array-like, optional
            best solution from the last iteration. If not given, best solution injection is disabled.
    
        Returns
        -------
        integer mutation vectors
        """
        m = lam
        n = len(xmean)
        if not self.flg_integer_mutation:
            return np.zeros((m, n))

        if bestx is None:
            bestx = xmean
        Rint = np.zeros((m, n), dtype=int)
        # Compute IR: masking vector: the components where an integer mutation will be applied
        IR = (2.0 * coordinate_std < self.stair_width)
        absIR = np.sum(IR)
        # compute lambda_int: number of solutions that the integer mutation will be applied
        if absIR == 0:
            lamint = 0
        elif absIR == n:
            lamint = m // 2
        else:
            lamint = min( m // 10 + absIR + 1, m // 2 - 1 )
        if lamint > 0:
            # compute Rp
            rnddim = np.nonzero(IR)[0][np.random.permutation(absIR)]
            Rp = np.zeros((lamint, n))
            for i in range(lamint):
                Rp[i, rnddim[i % absIR]] = 1
            # compute Rpp
            # Rpp = np.random.rand(lamint, n) >= 0.7 ** (1.0 / absIR)
            Rpp = np.zeros((lamint, n))
            Rpp[:, rnddim] = np.random.geometric(p=0.7 ** (1.0 / absIR), size=(lamint, absIR)) - 1
            # compute Iplusminus: sign-swiching (dim, lamint)-array for Rint, with each element being +-1
            #Iplusminus = np.random.randint(low=0, high=2, size=(lamint, n))
            Iplusminus = np.random.choice([-1, 1], size=(lamint, n))
            # compute Rint: integer mutation
            rndidx = np.random.permutation(m)
            Rint[rndidx[:lamint], :] = Iplusminus * (Rp + Rpp)
            #Rint[rndidx[lamint], stair_width > 0.0] = np.floor((bestx - xmean)[stair_width > 0.0] / stair_width[stair_width > 0.0])
            mask = np.where(self.stair_width > 0.0)[0]
            Rint[rndidx[lamint], mask] = np.floor(bestx[mask] / self.stair_width[mask]) - np.floor(xmean[mask] / self.stair_width[mask])
        return Rint * self.stair_width

    def mixed_integer_csa(self, weights, arz, coordinate_std):
        """Update Step-Size by CSA variation for Mixed Integer

        Correction for the flat (ineffective) dimension is implemented.
        
        Parameters
        ----------
        weights: 1d array-like
            recombination weights. Each value corresponds to each row of arz        
        arz : 2d array-like
            inversely_transformed steps (rows)
        coordinate_std : 1d array-like
            coordinate-wise standard deviation of gaussian

        Returns
        -------
        last_sig_change_fac (float) : multiplicant to update sigma
        hsig (bool) : flag to  check if the step-size is not going to be increased rapidly, normally 1.
        
        Note
        ----
        The correction is required since CSA suffer from ineffective axis. 
        For mixed integer problems the correction can be done explicitly, but 
        it is not possible in general. A preferred way would be to use other
        step-size control mechanisms such as TPA.
        """
        # update evolution path
        w = np.zeros(len(weights))
        w[weights > 0.0] = weights[weights > 0]
        muw = 1.0 / np.dot(w, w)
        dz = np.dot(w, arz)
        self.ps *= (1.0 - self.cs)
        self.ps += math.sqrt(self.cs * (2.0 - self.cs) * muw) * dz
        # update sigma
        self.hsig = np.linalg.norm(self.ps) < self.chiN * (1.5 + 1.0 / (len(self.ps) - 0.5))
        vecIsigma = (coordinate_std / math.sqrt(self.cs) >= 0.2 * self.stair_width)
        count_Isig = np.sum(vecIsigma)
        normps = np.linalg.norm(self.ps[vecIsigma])
        if count_Isig > 0:
            chin_mi = math.sqrt(count_Isig) * (1.0 - 1.0 / (4.0 * count_Isig) + 1.0 / (21.0 * count_Isig * count_Isig))
            self.last_sig_change_fac = math.exp((self.cs / self.ds) * (normps / chin_mi - 1.0))
        else:
            self.last_sig_change_fac = 1.0
        return self.last_sig_change_fac, self.hsig

    def std_correction(self, coordinate_std):
        """Std correction to prevent the coordinate-wise std to be too small

        Parameters
        coordinate_std : 1d array-like
            coordinate-wise standard deviation of gaussian

        Returns
        -------
        1d ndarray : multiplicant to update std
        """
        if self.correction_mode == 'sigma':
            return max(1.0, np.max(self.minstd / coordinate_std))
        elif self.correction_mode == 'coordinate_std':
            return np.fmax(1.0, self.minstd / coordinate_std)
        elif self.correction_mode == 'none':
            return 1
        else:
            raise ValueError(self.correction_mode)


class PeriodicVariableHandler(object):
    def __init__(self, lower, upper):
        """Periodic Variable Handler for CMA by Yamaguchi 2016 (in japanese)

        Parameters
        ----------
        lower, upper : array-like
            period = [lower, upper]. Set -np.inf, np.inf for non-periodic variables
        """
        self.lower = lower
        self.upper = upper

    def project_x(self, x):
        """Return x projected into the right period

        Parameters
        ----------
        x : array-like
            solution vector

        Return
        ------
        projected arx : 2d array-like

        Note
        ----
        Unnecessary to call this function if the objective function and the constraint functions
        are well-defined outside the right period.
        """
        period = self.upper - self.lower
        mask = period < np.inf
        projected = x.copy()
        if np.ndim(x) == 1:
            projected[mask] -= np.floor((x[mask] - self.lower[mask]) / period[mask]) * period[mask]
        elif np.ndim(x) == 2:
            projected[:, mask] -= np.floor((x[:, mask] - self.lower[mask]) / period[mask]) * period[mask]
        else:
            raise TypeError
        return projected

    def project_distribution(self, mean, coordinate_std):
        """project N(mean, cov)

        Parameters
        ----------
        mean : array-like
            mean vector
        coordinate_std : array-like
            sqrt of the diagonal elements of the covariance matrix. Note: sigma * diag(C)**0.5 for CMA

        Returns
        -------
        projected_mean (array-like) : projected mean
        projection_diag (array-like) : diagonal matrix to project the covariance matrix and others.

        Note
        ----
        1. The evolution path for rank-1 update should be projected by the return diagonal matrix.
        2. In case of full covariance matrix C, projection by D (i.e., D*C*D) can be done as
            C *= D
            C *= D[np.newaxis].T
            pc *= D
        3. In case that the mean vector is recorded for the later use (e.g., for evolution path update),
            projecting the mean vector will affect the code.
        """
        projected_mean = self.project_x(mean)
        D = np.fmin((self.upper - self.lower) / coordinate_std / 4.0, 1.0)
        return projected_mean, D


class MirroringBoxConstrainHandler(object):
    """Mirroring Box Constraint Handling
    f(x) = f(2*UB - x) if x > UB
    f(x) = f(2*LB - x) if x < LB

    Note
    ----
    If mirroring is applied, the corresponding variables become periodic,
    where the period is [LB - (UB - LB)/2, UB + (UB - LB)/2].
    Consider to apply PeriodicVariableHandler together with this class.

    See example for the usage
    """
    def __init__(self, lower, upper, mode='linear'):
        """

        Parameters
        ----------
        lower, upper : array-like
            if no bounds, put -np.inf (for lb) or np.inf (for ub)
        mode : str, optional
            ('linear', 'linquad')
        """
        assert mode in ('linear', 'linquad')
        self.lb = lower
        self.ub = upper
        self.mode = mode
        if self.mode == 'linquad':
            self.al = np.asarray([min( (self.ub[i] - self.lb[i]) / 2.0, (1.0 + np.abs(self.lb[i])) / 20.0 )
                                  if np.isfinite(self.lb[i]) else 1 for i in range(len(self.lb))])
            self.au = np.asarray([min( (self.ub[i] - self.lb[i]) / 2.0, (1.0 + np.abs(self.ub[i])) / 20.0 )
                                  if np.isfinite(self.ub[i]) else 1 for i in range(len(self.ub))])
        else:
            self.al = np.zeros(len(self.lb))
            self.au = np.zeros(len(self.ub))

    def get_mirrored_x(self, x):
        x_mirrored = np.array(x, copy=True)
        mask_l = (self.lb > -np.inf)
        mask_u = (self.ub < np.inf)
        mask_lu = mask_l * mask_u
        # step 1: projection of x into a base period [l - 2al - (u - l)/2, u + 2au + (u - l)/2]
        lb = self.lb[mask_l] - self.al[mask_l]
        ub = self.ub[mask_u] + self.au[mask_u]
        ll = 1.5 * self.lb[mask_lu] - 0.5 * self.ub[mask_lu] - 2.0 * self.al[mask_lu]
        uu = 1.5 * self.ub[mask_lu] - 0.5 * self.lb[mask_lu] + 2.0 * self.au[mask_lu]
        if np.ndim(x) == 1:
            x_mirrored[mask_lu] = x[mask_lu] - np.floor((x[mask_lu] - ll) / (uu - ll)) * (uu - ll)
            # assert np.all(ll <= x_mirrored[mask_lu]) and np.all(x_mirrored[mask_lu] < uu)
        elif np.ndim(x) == 2:
            x_mirrored[:, mask_lu] = x[:, mask_lu] - np.floor((x[:, mask_lu] - ll) / (uu - ll)) * (uu - ll)
        else:
            raise ValueError
        # step 2: mirroring
        if np.ndim(x) == 1:
            x_mirrored[mask_l] = lb + np.abs(lb - x_mirrored[mask_l])
            x_mirrored[mask_u] = ub - np.abs(ub - x_mirrored[mask_u])
        elif np.ndim(x) == 2:
            x_mirrored[:, mask_l] = lb + np.abs(lb - x_mirrored[:, mask_l])
            x_mirrored[:, mask_u] = ub - np.abs(ub - x_mirrored[:, mask_u])
        else:
            raise ValueError
        if self.mode is 'linquad':
            import warnings
            warnings.warn("This option is not yet tested. Check carefully and remove warning.")
            mask_al = (x_mirrored < self.lb + self.al)
            mask_au = (x_mirrored > self.ub - self.au)
            if np.ndim(x) == 1:
                x_mirrored[mask_al] = self.lb + (x_mirrored[mask_al] - lb) ** 2 / 4.0 / self.al[mask_al]
                x_mirrored[mask_au] = self.ub - (x_mirrored[mask_au] - ub) ** 2 / 4.0 / self.au[mask_au]
            elif np.ndim(x) == 2:
                x_mirrored[:, mask_al] = self.lb + (x_mirrored[:, mask_al] - lb) ** 2 / 4.0 / self.al[mask_al]
                x_mirrored[:, mask_au] = self.ub - (x_mirrored[:, mask_au] - ub) ** 2 / 4.0 / self.au[mask_au]
            else:
                raise ValueError
        return x_mirrored


class AdaptivePenaltyFunctionBoxConstraintHandler(object):

    def __init__(self, lower_bound, upper_bound, lambda_=1, mode='sakamoto17'):
        """Box Constraint Handling Based on Adaptive Penalty Function

        Two major variants are implemented
        hansen09 : The original version. Erratum and the modification presented in the author version of [Hansen 09] is implemented. 
        sakamoto17 : The improved version [Sakamoto 17]. 

        Parameters
        ----------
        lower_bound, upper_bound : ndarray
            lower and upper bound of each variable.
        lambda_ : int, optional
            lambda, population size, sample size per iteration
            To support adaptive population size, lambda_ should be the
            minimal number for the population size
        mode : str, defalt = 'sakamoto17'
            version ('hansen09', 'sakamoto17')
        """
        self.lb = lower_bound
        self.ub = upper_bound
        N = self.lb.shape[0]
        self.penalty = np.zeros(N)
        self.iqr_hist = deque(maxlen=20 + 3 * N // lambda_)
        self.flg_winit = False
        self.sqdm = 1.0
        self.meanviolation = 1.0
        assert mode in ('hansen09', 'sakamoto17'), 'Unsupported mode {}'.format(mode)
        self.mode = mode

    def repair(self, x, out=None):
        """Push each variable outside the feasible set onto the boundary

        Parameters
        ----------
        x : ndarray
            each row vector represents a point to be repaired
        out : ndarray, optional
            array where the output (repaired point) is stored

        Returns
        -------
        out : ndarray
            Repaired x. If `out` is passed, its reference is returned.
        """
        return np.clip(x, self.lb, self.ub, out)

    def check(self, x):
        """Check if x is feasible

        Parameters
        ----------
        x : ndarray (1D or 2D)
            a vector to be checked, or a matrix such that each row is a vector to be checked

        Returns
        -------
        ndarray (same dimension as x) : each value is -1 if x[i] < LB[i], 1 if x[i] > UB[i], 0 otherwise
        """
        return -1 * (x < self.lb) + (x > self.ub)

    def update_penalty_factor(self, lam, mueff, f_repaired, xmean, coordinate_std):
        # Update interquartile history
        q75, q50, q25 = np.percentile(f_repaired, [75, 50, 25])
        iqr = q75 - q25
        delf = iqr / np.mean(coordinate_std ** 2)

        if np.isfinite(delf) and delf > 0:
            self.iqr_hist.append(delf)
        else:
            self.iqr_hist.append(q50)
            print('Warning: Bad Initialization. An Invalid IQR observed.')
            print('delf = {}'.format(delf))

        # Penalty Weight Initialization
        if (not np.all(self.check(xmean) == 0) and
            (not self.flg_winit or len(self.iqr_hist) < 3)):
            # This initialization is performed at most twice (in first and second iterations)
            assert len(self.iqr_hist) > 0, 'No Feasible Solution. Try something like xmean = (lb + ub) / 2 and sigma = (ub - lb) / 6.'
            self.penalty[:] = 2.0 * self._effective_iqr_median(lam)
            self.flg_winit = True

        if self.flg_winit and len(self.iqr_hist) >= 3:

            # Increase the penalty coefficient
            dm = xmean - self.repair(xmean)
            N = xmean.shape[0]
            delm = np.abs(dm) / coordinate_std
            delth = 3 * max(1, math.sqrt(N) / mueff)
            dgam = min(1, mueff / (10 * N))
            if self.mode in ('hansen09'):
                self.penalty *= np.exp(
                    (dgam / 2) * np.tanh(np.fmax(0, delm - delth) / 3))
                self.penalty[self.penalty >
                             5 * np.median(self.iqr_hist)] *= math.exp(-dgam / 3)
            elif self.mode in ('sakamoto17'):
                mvio = np.fmax(0, delm - delth)
                self.penalty *= np.exp((dgam / 2) * np.tanh(mvio / 3))
                # Update the penalty coefficient based on IQR history
                scale = self._effective_iqr_median(lam) / np.mean(self.penalty)
                if 3.0 * scale < 1.0:
                    # Decrease
                    self.penalty[:] *= 3.0 * scale
                # elif scale > 1.0:                         #
                #     # Increase                            #   Change 2017.02.17
                #     self.penalty[mvio > 0] *= scale       #
                # Cinv_array = 1 / (sigma**2 * variance)
                # vCv_sqrt = np.sqrt(Cinv_array)
                # self.sqdm = np.sum((dm * Cinv_array / vCv_sqrt)**2)
                self.sqdm = np.sum(delm**2)

    def compute_penalty(self, x_raw, x_repaired, coordinate_std):
        """Compute Penalty

        Parameters
        ----------
        x_raw, x_repaired : ndarray
            each row of them is a point before and after the repair
        coordinate_std : ndarray (1D)
            coordinate-wise std of distribution
        """
        assert x_raw.shape == x_repaired.shape
        N = len(coordinate_std)
        if self.mode in ('hansen09'):
            return np.dot((x_repaired - x_raw)**2, self.penalty) / self.penalty.shape[0]
        elif self.mode in ('sakamoto17'):
            Cinv = 1.0 / coordinate_std ** 2
            vCv = Cinv.copy()
            self.meanviolation = np.dot(((x_raw-x_repaired) * Cinv.T)**2 / vCv.T, np.ones(N)) / N
            return np.dot((x_repaired - x_raw)**2, self.penalty) / self.penalty.shape[0]

    def _effective_iqr_median(self, lam):
        """Effective IQR Median

        Let L be the length of the IQR history (iqr_hist).
        L = 0 : error
        L <= 3 : median of iqr_hist
        L > 3 : let t be the current iteration.
            step 1. compute the median of three latest IQRs (medIQR)
            step 2. find the maximum T such that
                | log(IQR[-3-t]) - log(medIQR) | < log(5.0)
                for all t <= T <= maximum history length.
            step 3. compute the median of IQR[-3-T], ... , IQR[-1]

        Parameters
        ----------
        lam : int
            population size
        """
        N = self.lb.shape[0]
        if len(self.iqr_hist) == 0:
            raise RuntimeError
        elif len(self.iqr_hist) <= 3:
            return np.median(self.iqr_hist)
        else:
            arr_iqr_hist = np.asarray(self.iqr_hist)[:20+3*N//lam]
            med = np.median(arr_iqr_hist[-3:])
            tmp = np.abs(np.log(arr_iqr_hist[:-3]) - np.log(med))[::-1]
            i = np.where(tmp > np.log(5.0))[0]
            if len(i) > 0:
                i = i[0] + 3
            else:
                i = len(arr_iqr_hist)
            return np.median(arr_iqr_hist[-1:-i - 1:-1])


class AdaptivePenaltyFunctionLinearConstraintHandler(object):
    # Created by Sakamoto 2017.02.16, Revised by YA on 2017.10.13

    def __init__(self,
                 matA,
                 vecb,
                 method='vLCH',
                 dth_number=3):

        self.A = matA.reshape(1, len(matA)).astype(float) if matA.ndim == 1 else matA.astype(float)
        self.b = vecb.ravel().astype(float) if vecb.ndim == 2 else vecb.astype(float)
        # self.b[:] -= 1e-15
        N = self.A.shape[1]
        self.iqr_hist = deque(maxlen=20 + 3 * N)
        self.winit = False
        self.method = method
        self.dth_num = dth_number
        if method in ('vLCH'):
            self.penalty = np.zeros(self.A.shape[0])
        elif method in ('eLCH'):
            self.penalty = np.zeros(N)
        else:
            raise NameError('wrong lch-name')

    def violation(self, x):
        """

        Parameters
        ----------
        x : ndarray (1D or 2D)

        Returns
        -------

        """
        g_value = np.dot(x, self.A.T) - self.b
        return g_value

    # def repair(self, x, transform_inverse=None, epsrel=1e-8, epsabs=1e-14):
    #     N = x.shape[-1]
    #     y = x.reshape((-1, N))
    #     yfeas = np.empty(y.shape)
    #     invC = np.eye(N)
    #     if transform_inverse is not None:
    #         invC = transform_inverse(invC)
    #         invC = transform_inverse(invC.T)
    #     for i in range(y.shape[0]):
    #         yfeas[i] = compute_nearest_feasible(vec_x=y[i], mat_inv_c=invC, mat_a=self.A, vec_b=self.b)[0]
    #         # Force xfeas to be feasible
    #         while np.any(self.violation(yfeas[i]) > 0):
    #             dy = yfeas[i] - y[i]
    #             ndy = np.linalg.norm(dy)
    #             dy /= ndy
    #             yfeas[i] += (epsabs + epsrel * ndy) * dy
    #     yfeas.shape = x.shape
    #     return yfeas

    def repair(self, x, transform_inverse=None, init_eps=1e-14):
        if np.any(self.violation(x) > 0):
            N = x.shape[-1]
            y = x.reshape((-1, N))
            yfeas = y.copy()
            mat_inv_c = np.eye(N)
            if transform_inverse is not None:
                mat_inv_c = transform_inverse(transform_inverse(mat_inv_c).T)
            for i in range(y.shape[0]):
                eps = init_eps
                while np.any(self.violation(yfeas[i]) > 0):
                    yfeas[i] = compute_nearest_feasible(vec_x=y[i],
                                                        mat_inv_c=mat_inv_c,
                                                        mat_a=self.A,
                                                        vec_b=self.b-eps)[0]
                    eps *= 10
            return yfeas
        else:
            return x

    def update_penalty_factor(self, mueff, f_repaired, xmean, coordinate_std, transform):

        N = len(xmean)
        lam = len(f_repaired)
        damp = min(1, mueff / (10 * N))
        dth = (self.dth_num * max(1, N ** 0.5 / mueff))

        # Update interquartile history
        q75, q50, q25 = np.percentile(f_repaired, [75, 50, 25])
        iqr = q75 - q25

        delf = iqr / np.mean(coordinate_std ** 2)
        if np.isfinite(delf) and delf > 0:
            self.iqr_hist.append(delf)
        else:  # Add 2017.02.16
            self.iqr_hist.append(q50)
            print('Warning: Bad Initialization. An Invalid IQR observed.')
            print('delf = {}'.format(delf))

        # Prepare
        violation = self.violation(xmean)
        trimed = self._effective_iqr_median(lam)

        # Penalty coefficients Initialization
        # This initialization is performed at most twice (in first and second iterations)
        if np.any(violation > 0) and (not self.winit or len(self.iqr_hist) <= 2):
            self.penalty[:] = 2.0 * trimed
            self.winit = True

        # Update the penalty coefficient
        if self.winit and len(self.iqr_hist) > 2:
            if self.method == 'vLCH':
                # === v-LCH === #
                sqrtCv = transform(self.A)
                vCv = np.sum(sqrtCv ** 2, axis=1)
                edist = (violation / np.sqrt(vCv)) - dth
            elif self.method in ('eLCH'):
                # === e-LCH === #
                dm = xmean - self.repair(xmean)
                edist = (np.abs(dm) / coordinate_std) - dth
            else:
                raise ValueError
            # Increase penalty coefficient
            self.penalty *= np.exp((edist > 0) * np.tanh(edist / self.dth_num) * damp / 2.)
            # Decrease penalty coefficient
            scale = trimed / np.mean(self.penalty)
            if 3.0 * scale < 1.0:
                self.penalty[:] *= 3.0 * scale

    def compute_penalty(self, x_raw, x_repaired):
        """Compute Penalty

        Parameters
        ----------
        x_raw, x_repaired : ndarray
            each row of them is a point before and after the repair

        Returns
        -------
        ndarray (1D)
            array of penalty values
        """

        assert x_raw.shape == x_repaired.shape
        N = x_repaired.shape[-1]
        if (self.method == 'vLCH') or (not self.method):
            # === v-LCH === #
            return np.dot(np.fmax(np.dot(x_raw - x_repaired, self.A.T), 0) ** 2, self.penalty) / N
        elif self.method == 'eLCH':
            # === e-LCH === #
            return np.dot((x_raw - x_repaired) ** 2, self.penalty) / N

    def _effective_iqr_median(self, lam):
        """Effective IQR Median

        Let L be the length of the IQR history (iqr_hist).
        L = 0 : error
        L <= 3 : median of iqr_hist
        L > 3 : let t be the current iteration.
            step 1. compute the median of three latest IQRs (medIQR)
            step 2. find the maximum T such that
                | log(IQR[-3-t]) - log(medIQR) | < log(5.0)
                for all t <= T <= maximum history length.
            step 3. compute the median of IQR[-3-T], ... , IQR[-1]

        Parameters
        ----------
        lam : int
            population size
        """
        N = self.A.shape[1]
        if len(self.iqr_hist) == 0:
            raise RuntimeError
        elif len(self.iqr_hist) <= 3:
            return np.median(self.iqr_hist)
        else:
            arr_iqr_hist = np.asarray(self.iqr_hist)[:20+3*N//lam]
            med = np.median(arr_iqr_hist[-3:])
            tmp = np.abs(np.log(arr_iqr_hist[:-3]) - np.log(med))[::-1]
            i = np.where(tmp > np.log(5.0))[0]
            if len(i) > 0:
                i = i[0] + 3
            else:
                i = len(arr_iqr_hist)
            return np.median(arr_iqr_hist[-1:-i - 1:-1])


class UncertaintyHandler(object):
    """Uncertainty Handling Utilities based on [Hansen 09]"""
    def __init__(self,
                 N,
                 func=None,
                 r_lam=0.1,
                 theta=0.2,
                 alpha_sigma=None,
                 alpha_teval=1.5,
                 teval_min=1,
                 teval_max=np.inf,
                 eps_reev=1e-7,
                 cut_limit=np.inf,
                 cum_noise=1,
                 batch_evaluation=False,
                 additive_gauss=False):

        self.func = func
        self.r_lam = r_lam
        self.theta = theta
        self.alpha_sigma = 1 + 2 / (N + 10) if alpha_sigma is None else alpha_sigma
        self.alpha_teval = alpha_teval
        self.teval_min = teval_min
        self.teval_max = teval_max
        self.eps_reev = eps_reev
        self.cut_limit = cut_limit
        self.cum_noise = cum_noise
        self.noiselevel = 0
        self.teval = teval_min
        self.batch_evaluation = batch_evaluation
        self.additive_gauss = additive_gauss

    def eval(self, indi):
        """Evaluate averaged objective

        Parameters
        ----------
        indi : CmaSolution (or its derived class) instance

        Returns
        -------
        averaged objective function value (indi.arffeas = f(indi.x_to_eval))
        """
        arf = 0.
        if self.additive_gauss:
            arf = self.func(indi, n_average=self.teval)
        else:
            for t in range(self.teval):
                arf += self.func(indi)
            arf /= self.teval
        indi.arffeas = arf
        return arf

    @staticmethod
    def uncertainty_quantification(arf1, arf2, theta=0.2, cutlimit=np.Inf):
        """Uncertainty Quantification based on Hansen et al. (2009)"""

        # input argument verification
        lam = arf1.size
        lamreev = arf2.size
        assert (arf1.ndim == arf2.ndim ==
                1), "arf1 and arf2 should be numpy 1D-array"
        assert (lam >= lamreev), "arf1.size must be no less than arf2.size"

        if lam > lamreev:
            arf2 = np.hstack((arf2, arf1[lamreev:lam]))

        # compute rank change in rankDelta
        idx = np.hstack((arf1, arf2)).argsort(axis=0)
        ranks = idx.argsort(axis=0)
        rank_old = ranks[0:lamreev]
        rank_new = ranks[lam:lam + lamreev]
        rankDelta = rank_new - rank_old - np.sign(rank_new - rank_old)

        # compute the measurement
        X = np.tile(np.arange(2 * lam - 1), (lamreev, 1)).T
        Lim1 = np.tile(rank_new - (rank_new > rank_old), (2 * lam - 1, 1))
        Lim2 = np.tile(rank_old - (rank_old > rank_new), (2 * lam - 1, 1))
        pcntile1 = np.percentile(np.abs(X - Lim1), 50 * theta, axis=0)
        pcntile2 = np.percentile(np.abs(X - Lim2), 50 * theta, axis=0)
        s = 2 * np.abs(rankDelta) - pcntile1 - pcntile2

        # cut-off limit
        idx = np.abs(s) > cutlimit
        s[idx] = np.sign(s[idx]) * cutlimit

        # return
        return np.mean(s)

    def measure_noise(self, pop, func_mutate=None):
        # Re-Evaluation
        fpr = max(self.r_lam * len(pop), 2)
        lamreev = int(floor(fpr) + (rand(1) < (fpr - floor(fpr))))
        pop_re = [None] * lamreev
        arf_re = [None] * lamreev
        for i in range(lamreev):
            if func_mutate is not None:
                pop_re[i] = type(pop[i])(func_mutate(pop[i].x_to_eval, self.eps_reev))
            else:
                pop_re[i] = type(pop[i])(pop[i].x_to_eval)
            pop_re[i].x_to_eval = pop_re[i].arxfeas
            arf_re[i] = self.eval(pop_re[i])
        # update the noise level
        s = self.uncertainty_quantification(
            np.asarray([pop[i].arffeas for i in range(len(pop))]),
            np.asarray(arf_re), theta=self.theta, cutlimit=self.cut_limit)
        self.noiselevel += self.cum_noise * (s - self.noiselevel)
        return pop_re, arf_re, self.teval * lamreev

    @staticmethod
    def argsort(arf, arf_re):
        # input argument verification
        arf_ = np.asarray(arf)
        arf_re_ = np.asarray(arf_re)
        lam = arf_.size
        lamreev = arf_re_.size
        assert (arf_.ndim == arf_re_.ndim ==
                1), "arf1 and arf2 should be numpy 1D-array"
        assert (lam >= lamreev), "arf1.size must be no less than arf2.size"
        arf2 = np.hstack((arf_re_, arf_[lamreev:lam]))

        # compute rank change in rankDelta
        idx = np.hstack((arf_, arf2)).argsort(axis=0)
        ranks = idx.argsort(axis=0)
        rank_old = ranks[0:lamreev]
        rank_new = ranks[lam:lam + lamreev]
        rankDelta = rank_new - rank_old - np.sign(rank_new - rank_old)

        rank_avg = ranks[:lam] + ranks[lam:]
        rank_diff = np.zeros(lam, dtype=int)
        rank_diff[:len(rankDelta)] = np.abs(rankDelta)
        x = np.array(
            [i for i in zip(rank_avg, rank_diff, (arf_ + arf2) / 2)],
            dtype=np.dtype('i8, i8, f8')) #Modified by knishida on June 28, 2017
        # x = np.array(
        #     zip(rank_avg, rank_diff, (arf + arf2) / 2),
        #     dtype=np.dtype('i8, i8, f8'))
        return np.argsort(x)

    def uncertainty_handling(self, sigma):
        # Update teval or sigma
        if self.noiselevel > 0:
            if self.teval < self.teval_max:
                self.teval = int(
                    min(
                        max(self.teval + 1, self.alpha_teval * self.teval),
                        self.teval_max))
            else:
                sigma *= self.alpha_sigma
        elif self.noiselevel < 0:
            self.teval = int(
                max(
                    min(self.teval - 1, self.teval / self.alpha_teval),
                    self.teval_min))
        return self.teval, sigma


class PopSizeAdaptation(object):

    def __init__(self, lam_min=4, lam_max=np.inf, alpha=1.4, beta=0.5, dt=2.0, granularity=1):
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.psa_alpha = alpha
        self.psa_beta = beta
        self.psa_dt = dt
        self.psa_granularity = granularity
        self.pt_factor = 0.0
        self.pmnorm_list = []
        self.pcnorm_list = []
        assert lam_min % granularity == 0
    def register_cma(self, cma):
        if not hasattr(self, 'cma'):
            assert isinstance(cma, ABCCma)
            self.cma = cma
            self.N = self.cma.opts.N
            self.lam_dash = cma.popsize
            self.pxmean = np.zeros(cma.opts.N)
            self.pcov = np.zeros((cma.opts.N, cma.opts.N))
            self.old_M = self.cma.xmean.copy()
            self.old_invLt = self.cma.transform_inverse(np.eye(self.N))

    def update(self):
        # compute defference of the distribution parameters between before and after update
        new_Lt = self.cma.transform(np.eye(self.N))
        # L is supposed to be the square root, not cholesky
        delta_xmean = self.cma.xmean - self.old_M
        delta_xmean = np.dot(self.old_invLt.T, delta_xmean)
        delta_cov = np.dot(new_Lt, self.old_invLt)
        delta_cov = np.dot(delta_cov.T, delta_cov) - np.eye(self.N)
        # If L is not the square root, but cholesky, the following should work.
        # invDD, B = np.linalg.eigh(np.dot(self.old_invLt, self.old_invLt.T))
        # invSqrtC = np.dot(B, (B * np.sqrt(invDD)).T)
        # delta_xmean = self.cma.xmean - self.old_M
        # delta_xmean = np.dot(invSqrtC, delta_xmean)
        # delta_cov = np.dot(invSqrtC, new_Lt)
        # delta_cov = np.dot(delta_cov, delta_cov.T) - np.eye(self.N)

        # update the evolution path
        factor = self.cma.expected_update_snorm()
        pnormfac = math.sqrt(self.psa_beta * (2 - self.psa_beta) / factor)
        self.pxmean *= (1 - self.psa_beta)
        self.pxmean += pnormfac * delta_xmean
        self.pcov *= (1 - self.psa_beta)
        self.pcov += pnormfac * delta_cov

        # compute the norm of the evolution path
        self.pmnorm = np.sum(self.pxmean ** 2)
        self.pcnorm = np.sum(self.pcov ** 2) / 2
        self.pnorm = self.pmnorm + self.pcnorm
        self.pcnorm_list.append(self.pmnorm)
        self.pmnorm_list.append(self.pcnorm)
        # update the normarization factor of the evolution path
        self.pt_factor = (1 - self.psa_beta)**2 * self.pt_factor + (self.psa_beta * (2 - self.psa_beta))

        # update the population size
        self.lam_dash *= math.exp(1. / self.psa_dt * (self.pt_factor - self.pnorm / self.psa_alpha))
        self.lam_dash = min(max(self.lam_dash, self.lam_min), self.lam_max)
        lam = self.psa_granularity * int(round(self.lam_dash / self.psa_granularity))
        self.cma.popsize = lam  # do the whole job, including sigma normalization

        # record
        self.old_M = self.cma.xmean.copy()
        self.old_invLt = self.cma.transform_inverse(np.eye(self.N))


class InjectionHelper(object):
    def __init__(self, dim, clip_x=None, clip_m=None):
        """Injection Mechanism based on [Hansen 11]"""
        self.N = dim
        self.clip_x = clip_x if clip_x else math.sqrt(self.N) + 2.0 * self.N / (self.N + 2.0)
        self.clip_m = clip_m if clip_m else math.sqrt(2.0 *self.N) + 2.0 * self.N / (self.N + 2.0)

    def clip(self, a, b):
        assert a > 0 and b > 0
        return 1.0 if a > b else a / b

    def x_for_inject_direc(self, v, xmean, func_transform_inverse, random=True):
        """Generate x corresponding to the give direction v
            x = xmean + t * v / norm(transorm_inverse(v)),
        where t is norm(randn(N)) if random option is True, sqrt(N) otherwise.
        """
        if random:
            t = np.linalg.norm(np.random.randn(len(v)))
        else:
            t = np.sqrt(len(v))
        x = xmean + (t / np.linalg.norm(func_transform_inverse(v))) * v
        return x

    def x_for_inject_grad(self, v, xmean, func_transform, random=True):
        """Generate x corresponding to the give gradient c
            x = xmean + t * transform(transform(v)) / norm(transorm(v)),
        where t is norm(randn(N)) if random option is True, sqrt(N) otherwise.
        """
        if random:
            t = np.linalg.norm(np.random.randn(len(v)))
        else:
            t = np.sqrt(len(v))
        sqrtcv = func_transform(v)
        x = xmean + (t / np.linalg.norm(sqrtcv)) * func_transform(sqrtcv)
        return x

    def scaled_step_for_inject_x(self, x, xmean, func_transform_inverse):
        """Generate  a step y corresponding to the injected point x

        The step-length is scaled so that it will be comparable to the other solutions.
        """

        y = np.array(x, copy=True) - xmean
        z = func_transform_inverse(y)
        y *= self.clip(self.clip_x, np.linalg.norm(z))
        return y

    def scaled_step_for_inject_mean(self, x, xmean, muw, func_transform_inverse):
        """Generate a mean shift vector dm corresponding to the injected mean x

        The step-length is scaled so that it will be comparable to the standard cases.

        Note
        ----
        if a mean vector is injected, set cmu = 0 and perform the other update.
        """
        dm = x - xmean
        dm *= self.clip(self.clip_m, np.sqrt(muw) * np.linalg.norm(func_transform_inverse(dm)))
        return dm


class FuncWrapper:
    """Function Wrapper Class for Benchmark"""
    def __init__(self, func, batcheval=False):
        """
        Parameters
        ----------
        func : callable
            It takes numpy.ndarray and returns to the objective function values.
        batcheval : bool
            batch evaluation is available if it is true
        """
        self._func = func
        self._batcheval = batcheval
    def __call__(self, cma_solution, *args, **kwargs):
        """
        Parameters
        ----------
        cma_solution : (list of) CmaSolution (or its derived) class instance(s),

        Returns
        -------
        (list of) cma_solution.arffeas (over-written)
        """
        if isinstance(cma_solution, list):
            if self._batcheval:
                xarr = np.asarray([sol.x_to_eval for sol in cma_solution])
                farr = self._func(xarr, *args, **kwargs)
                for i in range(len(cma_solution)):
                    cma_solution[i].arffeas = farr[i]
                return list(farr)
            else:
                raise NotImplementedError
        else:
            assert isinstance(cma_solution, CmaSolution), "type(cma_solution) == {}".format(type(cma_solution))
            cma_solution.arffeas = self._func(cma_solution.x_to_eval, *args, **kwargs)
            return cma_solution.arffeas


class ABCCma(ABCIterativeAlgorithm):
    """Interface for Variations of CMA-ES

    The main purpose of this interface is to support the functionarities
    implemented in this file `cmaes_util` for any variations of CMA-ES.
    """

    def __init__(self, opts):
        """Constructor of `ABCCma`

        Parameters
        ----------
        opts : ABCCmaOption
        """
        super(ABCCma, self).__init__(opts)
        assert hasattr(self.opts, 'N')
        assert hasattr(self.opts, 'lam')
        # Tolerance Checker
        if self.opts.check_tolf_iter > 0:
            self.tolf_checker = TolfChecker(self.opts.check_tolf_iter)
            self.tolf = self.opts.check_tolf
            self.tolfrel = self.opts.check_tolfrel
        if self.opts.check_tolx_iter > 0:
            self.tolx_checker = TolxChecker(self.opts.N, self.opts.check_tolx_iter)
            self.tolx = self.opts.check_tolx
            self.tolxrel = self.opts.check_tolxrel
        # Mixed Integer
        if self.opts.mixed_integer_stair_width is not None:
            self.mixed_integer_handler = MixedIntegerHandler(self.opts.mixed_integer_stair_width,
                                                             minstd=self.opts.mixed_integer_minstd,
                                                             correction_mode=self.opts.mixed_integer_correction_mode,
                                                             flg_integer_mutation=self.opts.mixed_integer_flg_integer_mutation)
        # Periodic Variable
        if self.opts.periodic_lower is not None and self.opts.periodic_upper is not None:
            self.periodic_handler = PeriodicVariableHandler(
                lower=self.opts.periodic_lower,
                upper=self.opts.periodic_upper)
        # Box Constraint
        if self.opts.lower_bound is not None or self.opts.upper_bound is not None:
            if self.opts.box_constraint_handler in ('mirror'):
                self.box_constraint_handler = MirroringBoxConstrainHandler(
                    lower=self.opts.lower_bound,
                    upper=self.opts.upper_bound)
                if not hasattr(self, 'periodic_handler'):
                    raise UserWarning('Periodic Helper with period (1.5 * LB - 0.5 * UB, 1.5 * UB - 0.5 * LB) is desired to be used with Mirroring BCH.')
            elif self.opts.box_constraint_handler in ('hansen09', 'sakamoto17'):
                self.box_constraint_handler = AdaptivePenaltyFunctionBoxConstraintHandler(
                    lower_bound=self.opts.lower_bound,
                    upper_bound=self.opts.upper_bound,
                    mode=self.opts.box_constraint_handler)
            else:
                raise NotImplementedError('Unsupported value {} for opts.box_constraint_handler'.format(self.opts.box_constraint_handler))
        # Linear Constraint
        if self.opts.lch_b is not None and self.opts.lch_A is not None:
            if self.opts.lch_method in ('vLCH', 'eLCH'):
                # TODO: AdaptivePenaltyFunctionBoxConstraintHandler should not be used at the same time
                self.linear_constraint_handler = AdaptivePenaltyFunctionLinearConstraintHandler(
                    matA=self.opts.lch_A,
                    vecb=self.opts.lch_b,
                    method=self.opts.lch_method)
            else:
                raise NotImplementedError(
                    'Unsupported value {} for opts.lch_method'.format(self.opts.lch_method))
        # Quantifiable/Relaxable/Simulation-based/Known (QRSK) Constraint
        # if self.opts.qrsk_constraint is not None:
            # if self.opts.qrsk_constraint_handler == "augmented_lagrangian":
            #     self.qrsk_constraint_handler = AugmentedLagrangianConstraintHandler(*self.opts.qrsk_constraint)
            # else:
                # raise NotImplementedError(
                #     'Unsupported value {} for opts.qrsk_constraint_handler'.format(self.opts.qrsk_constraint_handler))
        # Uncertainty
        if self.opts.uncertainty:
            if (hasattr(self, 'box_constraint_handler') and isinstance(self.box_constraint_handler, AdaptivePenaltyFunctionBoxConstraintHandler)):
                assert self.opts.uh_eps_reev == 0, "self.opts.uh_eps_reev must be 0 if combined with `AdaptivePenaltyFunctionBoxConstraintHandler`"
            self.uncertainty_handler = UncertaintyHandler(
                N=self.opts.N,
                r_lam=self.opts.uh_r_lam,
                theta=self.opts.uh_theta,
                alpha_sigma=self.opts.uh_alpha_sigma,
                alpha_teval=self.opts.uh_alpha_teval,
                teval_min=self.opts.uh_teval_min,
                teval_max=self.opts.uh_teval_max,
                eps_reev=self.opts.uh_eps_reev,
                cut_limit=self.opts.uh_cut_limit,
                cum_noise=self.opts.uh_cum_noise,
                batch_evaluation=self.opts.batch_evaluation,
                additive_gauss=self.opts.additive_gauss)
        # Population Size Adaptation.  Note: do not forget `register` cma
        if self.opts.psa:
            self.popsize_adaptation = PopSizeAdaptation(lam_min=self.opts.psa_lam_min,
                                                        lam_max=self.opts.psa_lam_max,
                                                        alpha=self.opts.psa_alpha,
                                                        beta=self.opts.psa_beta,
                                                        dt=self.opts.psa_dt,
                                                        granularity=self.opts.psa_granularity)
        # Injection
        self.injection_helper = InjectionHelper(
            dim=self.opts.N,
            clip_x=self.opts.injection_clip_x,
            clip_m=self.opts.injection_clip_m)
        self.best_from_last = None
        self.best_so_far = None
        self.arxinj = []
        # Solution Class
        self.cls_solution = self.opts.cls_solution
        # Record
        self.fbest = np.inf
        self.neval = 0


    @property
    @abstractmethod
    def popsize(self):
        """int : population size

        Notes
        -----
        Setting the popsize should update all the relevant parameters such as
        recombination weights and learning rates.

        If necessary, the step-size normalization should be done here.
        """
        pass

    @popsize.setter
    @abstractmethod
    def popsize(self, pop_size):
        pass

    @property
    @abstractmethod
    def xmean(self):
        """:obj:`numpy.ndarray` (1 dimensional) : Mean vector"""
        pass

    @xmean.setter
    @abstractmethod
    def xmean(self, x):
        pass

    @property
    @abstractmethod
    def coordinate_std(self):
        """:obj:`numpy.ndarray` (1 dimensional) : coordinate-wise standard deviation"""
        pass

    @abstractmethod
    def multiply_covariance_by(self, d):
        """Multiply the covariance matrix by the diagonal matrix

        Parameter
        ---------
        d : float or 1D array
            a scalar or a vector of elements of a diagonal matrix

        Note
        ----
        In case of full covariance matrix, one might need to re-compute
        the eigen decomposition since d * C * d may be sufficiently different
        from C. 

        Naive way is as follows:
        C *= d
        C *= d[np.newaxis].T
        """
        pass

    @abstractmethod
    def transform(self, x):
        """Transform the covariance of a random vector from I to S = s**2*C
        
        Parameters
        ----------
        x : 1D or 2D array-like
            vector(s) with zero mean and identity covariance

        Returns
        -------
        xt (same type as x) : transformed to make covariance S

        Note
        ----
        Naive way is as follows
        DD, B = np.linalg.eigh(C)
        D = np.sqrt(D)
        sqrtS = np.dot(B * (sigma * D), B.T)
        return np.dot(x, sqrtS)
        """
        pass

    @abstractmethod
    def transform_inverse(self, x):
        """Transform the covariance of a random vector from S = s**2*C to I

        Parameters
        ----------
        x : 1D or 2D array-like
            vector(s) with zero mean and covariance S

        Returns
        -------
        xt (same type as x) : transformed to make covariance I

        Note
        ----
        Naive way is as follows
        DD, B = np.linalg.eigh(C)
        D = np.sqrt(D)
        invsqrtS = np.dot(B / (sigma * D), B.T)
        return np.dot(x, invsqrtS)
        """
        pass

    @abstractmethod
    def _update(self, pop):
        """Update the distribution parameters and others

        Parameters
        ----------
        pop : list of CmaSolution (or its derived class) instances
        """
        pass

    @property
    def measure(self):
        return self.neval

    @property
    def criterion(self):
        return self.fbest

    @property
    def recommendation(self):
        return self.xmean

    def expected_update_snorm(self):
        """Compute the expected square norm of the normalized parameter update

        Returns
        -------
        float
            expected square norm of the normalized parameter update
        """
        raise NotImplementedError('Necessary for PopSizeAdaptation')

    def mutate(self, x, mut_strength=1.0):
        """Mutate x with a given mutation strength

        Parameters
        ----------
        x : 1D or 2D array-like
            vector(s) to be mutated
        mut_strength : float, default = 1.0
            mutation strength

        Returns
        -------
        mutants (same shape as x)
        """
        x_ = np.asarray(x)
        z = np.random.randn(*x_.shape)
        return x_ + mut_strength * self.transform(z)

    def repair(self, x):
        """Repair Solution

        Parameters
        ----------
        x : 1D or 2D array-like
            vector(s) to be repaired
        Returns
        -------
        repaired vectors (same shape as x) to be evaluated
        """
        # Linear Constraint Support
        if hasattr(self, 'linear_constraint_handler'):
            if isinstance(self.linear_constraint_handler, AdaptivePenaltyFunctionLinearConstraintHandler):
                return self.linear_constraint_handler.repair(x)
            else:
                self.raise_value_error('linear_constraint_handler')
        # Box Constraint Support
        elif hasattr(self, 'box_constraint_handler'):
            if isinstance(self.box_constraint_handler, AdaptivePenaltyFunctionBoxConstraintHandler):
                return self.box_constraint_handler.repair(x)
            elif isinstance(self.box_constraint_handler, MirroringBoxConstrainHandler):
                return self.box_constraint_handler.get_mirrored_x(x)
            else:
                self.raise_value_error('box_constraint_handler')
        else:
            return x

    def inject_solution(self, pop):
        """Inject Solutions

        Parameters
        ----------
        pop : list of CmaSolution (or its derived class) instances
        """
        self.arxinj += pop

    def _check(self):
        """Inherited from ABCIterativeAlgorithm"""
        is_satisfied, condition = super(ABCCma, self)._check()
        if not is_satisfied:
            is_satisfied, condition = self._check_tolerance()
        return is_satisfied, condition

    def _onestep(self):
        """Inherited from ABCIterativeAlgorithm"""
        assert hasattr(self, 'func')
        # Population Size Adaptation
        if hasattr(self, 'popsize_adaptation'):
            self.popsize_adaptation.register_cma(self)

        # Best Solution Injection
        if self.opts.injection_mode == 'best' and self.best_so_far is not None:
            self.inject_solution([self.best_so_far])

        # Augmented Lagrangian
        # if (hasattr(self, 'qrsk_constraint_handler') and
        #         isinstance(self.qrsk_constraint_handler, AugmentedLagrangianConstraintHandler)):
        #     xmean_solution = self.cls_solution(self.xmean)
        #     xmean_solution.arxfeas[:] = self.repair(xmean_solution.arxraw + xmean_solution.arxmut)
        #     xmean_solution.x_to_eval[:] = xmean_solution.arxfeas[:]
        #     self.qrsk_constraint_handler.update(xmean_solution,
        #                                         self.evaluate_objective(xmean_solution))

        # Sampling and Evaluation
        pop, nevals = self.sample_and_evaluate(self.opts.N, self.popsize, parallel=self.opts.parallel)
        assert len(nevals) == 1, 'counting the numbers of calls of multiple functions not supported yet.'

        # Uncertainty Support
        if hasattr(self, 'uncertainty_handler'):
            # TODO: Combination with Constraint Handler not yet supported
            pop_re, arf_re, neval_re = self.uncertainty_handler.measure_noise(pop, self.mutate)
            self.neval += neval_re + self.uncertainty_handler.teval * nevals[0]
            teval, sigmafac = self.uncertainty_handler.uncertainty_handling(1.0)
            self.multiply_covariance_by(sigmafac)
            idx = self.uncertainty_handler.argsort(np.asarray([indi.arffeas for indi in pop]),
                                                   np.asarray([indi.arffeas for indi in pop_re]))
            for i in range(len(pop)):
                # In this case `arfit` is the ranking starting from 0
                pop[idx[i]].arfit = i
        else:
            self.neval += nevals[0]
        # Update
        self._update(pop)
        # Population Size Adaptation
        if hasattr(self, 'popsize_adaptation'):
            self.popsize_adaptation.update()
        # Mixed Integer
        if hasattr(self, 'mixed_integer_handler'):
            std = self.mixed_integer_handler.std_correction(self.coordinate_std)
            self.multiply_covariance_by(std)
        # Periodic
        if hasattr(self, 'periodic_handler'):
            m, D = self.periodic_handler.project_distribution(self.xmean, self.coordinate_std)
            self.multiply_covariance_by(D)
            self.xmean = m
        # Finalize
        self.pop = pop
        self.fbest = np.min([indi.arfit for indi in pop])
        self.best_from_last = pop[np.argmin([indi.arffeas for indi in pop])]
        if self.best_so_far is None or self.best_so_far.arffeas >= self.best_from_last.arffeas:
            self.best_so_far = self.best_from_last

    def sample_candidate(self, pop):
        """Generate Candidate Solutions 

        The following attributes are filled
        arxraw : mutant vector
        arxmut : integer mutation vector
        arxfeas : repaired solution (originally arxraw + arxmut)

        Parameters
        ----------
        pop : list of CmaSolution (or its derived class) instances
            list of (empty) candidate solutions
        
        Returns
        -------
        pop : list of filled candidate solutions

        Support
        -------
        * Mirrored Mutation
        * Integer Mutation
        * Repair Operator
        * Periodic Variable

        See
        ---
        `MixedIntegerHandler`, `mutate`, `repair`
        """
        n_sample = len(pop)
        # arxraw
        for idx in range(n_sample):
            indi = pop[idx]
            if not self.opts.mirror or idx % 2 == 0:
                indi.arxraw[:] = self.mutate(self.xmean, mut_strength=1.0)
            else:
                indi.arxraw[:] = 2.0 * self.xmean - pop[idx - 1].arxraw
        # arxmut
        if hasattr(self, 'mixed_integer_handler'):
            if not isinstance(self.mixed_integer_handler, MixedIntegerHandler):
                self.raise_value_error('mixed_integer_handler')
            else:
                bestx = self.best_from_last.arxfeas if self.best_from_last else None
                arxmut = self.mixed_integer_handler.integer_mutation(n_sample, self.xmean, self.coordinate_std, bestx=bestx)
        else:
            arxmut = np.zeros((n_sample, len(pop[0].arxmut)))
        # arxfeas and x_to_eval
        for idx in range(n_sample):
            indi = pop[idx]
            indi.arxmut[:] = arxmut[idx]
            indi.arxfeas[:] = self.repair(indi.arxraw + indi.arxmut)
            # Periodic Support
            if hasattr(self, 'periodic_handler'):
                indi.x_to_eval[:] = self.periodic_handler.project_x(indi.arxfeas)
            else:
                indi.x_to_eval[:] = np.asarray(indi.arxfeas)
        return pop

    def evaluate_objective(self, pop, parallel=False):
        """Compute objective function value

        pop[i].arffeas <- f(pop[i].x_to_eval)

        Parameters
        ----------
        pop : list of CmaSolution
            solutions to be evaluated
        parallel : bool, optional
            evaluation will be parallelized if it is True.

        Returns
        list of objective values (float)
        """
        assert hasattr(self, 'func')
        # Uncertainty Handling Support
        if hasattr(self, 'uncertainty_handler'):
            self.uncertainty_handler.func = self.func
            fobj = self.uncertainty_handler.eval
            # def fobj(x):
            #     assert isinstance(x, CmaSolution)
            #     if self.opts.additive_gauss:
            #         f = self.func(x, n_average=self.uncertainty_handler.teval)
            #     else:
            #         f = 0.0
            #         for t in range(self.uncertainty_handler.teval):
            #             f += self.func(x)
            #         f /= self.uncertainty_handler.teval
            #     return f
        else:
            fobj = self.func

        if isinstance(pop, CmaSolution):
            return fobj(pop)
        elif parallel:
            # Parallel Evaluation
            flist = fobj(pop)
            # TODO: Below Not working. f-values are not set to pop
            # lam = len(pop)
            # n_cpu = multiprocessing.cpu_count()
            # lam_cpu = np.linspace(0, lam, num=n_cpu+1, endpoint=True, dtype=int)
            # subpop_list = [pop[lam_cpu[i]:lam_cpu[i+1]] for i in range(n_cpu)]
            # with multiprocessing.Pool(processes=n_cpu) as pool:
            #     flist_list = pool.map(self.evaluate_objective, subpop_list)
            # flist = []
            # for fsublist in flist_list:
            #     flist += fsublist
        else:
            # Serial Evaluation
            flist = [fobj(indi) for indi in pop]
        return flist


    def sample_and_evaluate(self, dim, lam, parallel=False):
        """Sample and Evaluate
        
        Parameters
        ----------
        dim : int
            dimension
        lam : int
            number of candidate solutions
        mueff : float, optional
            1 / sum of (cm * wi) ** 2
            cm : learning rate for mean
            wi : recombination weights used for m-update
            It is only necessary for `AdaptivePenaltyFunctionBoxConstraintHandler`
        parallel : bool, optional
            parallel evaluation will be performed if it is True

        Returns
        -------
        list of CmaSolution (or its derived class) instances : solution set including injected solutions
        1D array-like : number of function calls
        """
        n_injected = len(self.arxinj)
        lam_new = lam - n_injected
        # Execution
        total_nevals = np.zeros(1)
        # Resampling Support
        if self.opts.resampling > 1:
            pop = []
            for i in range(lam_new):
                idx_feas, resampled, nevals = resample(dim=dim,
                                                       sample_func=self.sample_candidate,
                                                       func_list=[partial(self.evaluate_objective, parallel=parallel)],
                                                       non_positive_indices=None,
                                                       maxresample=self.opts.resampling,
                                                       cls=self.cls_solution)
                resampled[-1].arffeas = resampled[-1].values[0]
                pop.append(resampled[-1])
                if len(nevals) > 1:
                    # TODO: nevals for constraint not supported yet
                    raise NotImplementedError('Only resampling for objective is supported.')
                total_nevals += np.asarray(nevals)
        else:
            pop = [self.cls_solution(np.zeros(dim) * np.nan) for i in range(lam_new)]
            pop = self.sample_candidate(pop)
            self.evaluate_objective(pop, parallel=parallel)
            total_nevals += np.asarray([lam_new])
        # Evaluate Injected
        for indi in self.arxinj:
            if np.isnan(indi.arffeas):
                indi.arxfeas[:] = self.repair(indi.arxraw + indi.arxmut)
                # Periodic Support
                if hasattr(self, 'periodic_handler'):
                    indi.x_to_eval[:] = self.periodic_handler.project_x(indi.arxfeas)
                else:
                    indi.x_to_eval[:] = np.asarray(indi.arxfeas)
                self.evaluate_objective(indi)
                total_nevals[0] += 1
            pop += [indi]
        self.arxinj = []
        # Penalty
        # Linear Constraint Support
        if (hasattr(self, 'linear_constraint_handler') and
                isinstance(self.linear_constraint_handler, AdaptivePenaltyFunctionLinearConstraintHandler)):
            assert hasattr(self, 'mueff')
            self.linear_constraint_handler.update_penalty_factor(self.mueff,
                                                                 [indi.arffeas for indi in pop],
                                                                 self.xmean,
                                                                 self.coordinate_std,
                                                                 self.transform)
            for indi in pop:
                indi.arpen = self.linear_constraint_handler.compute_penalty(indi.arxraw + indi.arxmut, indi.arxfeas)
        elif (hasattr(self, 'box_constraint_handler') and
                  isinstance(self.box_constraint_handler, AdaptivePenaltyFunctionBoxConstraintHandler)):
            assert hasattr(self, 'mueff')
            self.box_constraint_handler.update_penalty_factor(lam, self.mueff, [indi.arffeas for indi in pop], self.xmean, self.coordinate_std)
            for indi in pop:
                indi.arpen = self.box_constraint_handler.compute_penalty(indi.arxraw + indi.arxmut, indi.arxfeas, self.coordinate_std)
        # elif (hasattr(self, 'qrsk_constraint_handler') and
        #           isinstance(self.qrsk_constraint_handler, AugmentedLagrangianConstraintHandler)):
        #     for indi in pop:
        #         indi.arpen = self.qrsk_constraint_handler(indi)
        elif (hasattr(self, 'stochastic_constraint_handler') and
              isinstance(self.stochastic_constraint_handler, StochasticConstraintHandler)):
            pop = self.stochastic_constraint_handler(pop)
        else:
            for indi in pop:
                indi.arpen = 0.0
        for indi in pop:
            indi.arfit = indi.arffeas + indi.arpen
        return pop, total_nevals

    def _check_tolerance(self):
        """Check Tolerance"""
        is_satisfied = False
        condition = ''
        if hasattr(self, 'tolf_checker'):
            self.tolf_checker.update([indi.arfit for indi in self.pop])
            if not is_satisfied and self.tolf_checker.check_absolute(self.tolf):
                is_satisfied = True
                condition = 'tolf'
            if not is_satisfied and self.tolf_checker.check_relative(self.tolfrel):
                is_satisfied = True
                condition = 'tolfrel'
            if not is_satisfied and self.tolf_checker.check_flatarea():
                is_satisfied = True
                condition = 'flatarea'
        if hasattr(self, 'tolx_checker'):
            self.tolx_checker.update([indi.arxraw for indi in self.pop])
            if not is_satisfied and self.tolx_checker.check_absolute(self.tolx):
                is_satisfied = True
                condition = 'tolx'
            if not is_satisfied and self.tolx_checker.check_relative(self.tolxrel):
                is_satisfied = True
                condition = 'tolxrel'
        return is_satisfied, condition
    
    def raise_value_error(self, attribute_name):
        raise ValueError("Unsupported type {} for self.{}".format(type(getattr(self, attribute_name)), attribute_name))


class ABCCmaOption(IterativeAlgorithmOption):
    def __init__(self,
                 # Termination
                 check_tolf_iter='20 + N // lam # tolf is disabled if 0',
                 check_tolf='check_target / 1e3',
                 check_tolfrel='1e-12',
                 check_tolx_iter='20 + N // lam # tolx is disabled if 0',
                 check_tolx='1e-10',
                 check_tolxrel='1e-12',
                 # Mixed Integer
                 mixed_integer_stair_width='None # stair width',
                 mixed_integer_minstd='0 # minimum coordinate_std',
                 mixed_integer_correction_mode='"sigma" # correction mode',
                 mixed_integer_flg_integer_mutation='True # flag for integer mutation',
                 # Uncertainty
                 uncertainty='False',
                 uh_r_lam=0.1,
                 uh_theta=0.2,
                 uh_alpha_sigma=None,
                 uh_alpha_teval=1.5,
                 uh_teval_min=1,
                 uh_teval_max=np.inf,
                 uh_eps_reev=1e-7,
                 uh_cut_limit=np.inf,
                 uh_cum_noise=1,
                 batch_evaluation=repr(False),
                 additive_gauss=repr(False),
                 # Resampling
                 resampling='1 # number of resampling',
                 # Periodic
                 periodic_lower='None # lower bound of periodic variables',
                 periodic_upper='None # upper bound of periodic variables',
                 # Box Constraint
                 lower_bound='None # None or ndarray',
                 upper_bound='None # None or ndarray',
                 box_constraint_handler='"mirror" # ("mirror", "hansen09", "sakamoto17")',
                 # Linear Constraint Handler
                 lch_A='None # A in np.dot(A, x) <= b',
                 lch_b='None # b in np.dot(A, x) <= b ',
                 lch_method='"vLCH" # ("vLCH", "eLCH")',
                 # Quantifiable/Relaxable/Simulation-based/Known (QRSK) Constraint Handler
                #  qrsk_constraint='None # tuple of constraint functions',
                #  qrsk_constraint_handler='"augmented_lagrangian"',
                 # Injection Helper
                 injection_clip_x='None # float of clip for x injection',
                 injection_clip_m='None # float of clip for mean injection',
                 injection_mode='None # "best" or "initial" or None',
                 # Mirrored Sampling
                 mirror='False # (bool) Mirrored Sampling Mode',
                 # Parallel Evaluation
                 parallel='False # (bool) Parallel Function Evaluation',
                 # Population Size Adaptation
                 psa='False # (bool) Population Size Adaptation Mode',
                 psa_lam_min='4 # minimum population size',
                 psa_lam_max='np.inf # maximum population size',
                 psa_alpha='1.4 # threshold',
                 psa_beta='0.5 # cumulation factor for normalized update',
                 psa_dt='1/float(psa_beta) # dampling for popsize update',
                 psa_granularity='1 # guranularity of adapted popsize',
                 # Solution Class
                 cls_solution=CmaSolution,
                 # Others
                 **kwargs):
        super(ABCCmaOption, self).__init__(**kwargs)
        self.setattr_from_local(locals())


class Cma(ABCCma):
    """Covariance Matrix Adaptation Evolution Strategy for Continuous Optimization
    """
    @property
    def popsize(self):
        return self._lam

    @popsize.setter
    def popsize(self, pop_size):
        self._lam = pop_size
        opts = self.opts_original
        opts.lam = self._lam
        parsed = opts.parse()
        self._w = parsed.w            # TODO:
        self.mueff = parsed.mueff     # TODO:
        self.cm = parsed.cm
        self.cc = parsed.cc
        self.cone = parsed.cone
        self.cmu = parsed.cmu
        self.ssa = parsed.ssa
        if self.ssa == 'TPA':
            self.cs = parsed.tpa_cs
            self.ds = parsed.tpa_ds
            if not hasattr(self, 'ps'):
                self.ps = 0
        elif self.ssa == 'CSA':
            self.cs = parsed.csa_cs
            self.ds = parsed.csa_ds
            if not hasattr(self, 'ps'):
                self.ps = np.zeros(self.N)
                self.ps_factor = 0.0
        elif self.ssa == 'NCSA':
            self.cs = parsed.ncsa_cs
            self.ds = parsed.ncsa_ds
            if not hasattr(self, 'ps'):
                self.ps = np.zeros(self.N)
        elif self.ssa == 'HSA':
            self.cs = parsed.hsa_cs
            self.ds = parsed.hsa_ds
            self.asig = parsed.hsa_as
            if not hasattr(self, 'ps'):
                self.ps = np.zeros(self.N)
        elif self.ssa == 'MICSA':
            self.mixed_integer_handler.cs = parsed.csa_cs
            self.mixed_integer_handler.ds = parsed.csa_ds
        elif self.ssa == 'NoSSA':
            pass
        else:
            raise NotImplementedError('ssa = ' + self.ssa +
                                      ' is not implemented.')
        self._w = pre_weight(self.N, self._lam, self._w, self.cone, self.cmu)
        if hasattr(self, '_last_alpha'):
            # update step-size
            next_alpha = sigma_normalization_factor(self.N, self._w, self.cm)
            self.sigma *= next_alpha / self._last_alpha
            self._last_alpha = next_alpha
        else:
            self._last_alpha = sigma_normalization_factor(self.N, self._w, self.cm)

    @property
    def xmean(self):
        return self._xmean

    @xmean.setter
    def xmean(self, x):
        self._xmean[:] = x

    @property
    def coordinate_std(self):
        return self.sigma * np.sqrt(np.diag(self.cov))

    def multiply_covariance_by(self, d):
        s = math.exp(np.mean(np.log(d)))
        nd = d / s
        if np.any(nd != 1):
            self.cov *= nd[np.newaxis].T
            self.cov *= nd
            self.sigma *= s
        else:
            self.sigma *= s

    def transform(self, x):
        return np.dot(np.asarray(x), self.sqrtC) * self.sigma

    def transform_inverse(self, x):
        return np.dot(np.asarray(x), self.invsqrtC) / self.sigma

    def _decompose(self, det_normalize=False):
        """Decompose the covariance matrix and update relevant parameters"""
        DD, self.B = np.linalg.eigh(self.cov)
        self.D = np.sqrt(DD)
        if not (self.D.max() < self.D.min() * 1e7):
            raise RuntimeError('Condition number > 1e7 or nan appears.')
        if det_normalize:
            gamma = np.exp(np.mean(np.log(self.D)))
            self.D /= gamma
            self.pc /= gamma
            self.cov /= gamma ** 2
        self.sqrtC = np.dot(self.B * self.D, self.B.T)
        self.invsqrtC = np.dot(self.B / self.D, self.B.T)
        self.eigneval = self.measure
        return gamma if det_normalize else 1

    def _update(self, pop):
        """Update the distribution parameters and others

        See Also : `ABCCma`

        Parameters
        ----------
        pop : list of CmaSolution (or its derived class) instances
        """
        idx = np.argsort([indi.arfit for indi in pop])
        sary = np.asarray([(pop[i].arxraw - self.xmean) / self.sigma for i in idx])
        sarx = np.asarray([pop[i].arxraw + pop[i].arxmut for i in idx])
        lam = len(pop)
        # idx = pop.idx
        # sary = (pop.arxraw[idx] - self.xmean) / self.sigma
        # sarx = pop.arxraw[idx] + pop.arxmut[idx]
        # lam = pop.arxraw.shape[0]
        w = weight(pop, self.xmean, self.transform_inverse,
                   base_weight=self._w, cone=self.cone, cmu=self.cmu)
        mueff = np.sum(w[w > 0]) / np.sum(w[w > 0] ** 2) if np.any(w > 0) else 0
        self.w = w  # used for PSA

        # TODO: break if mueff == 0 (no positive update performed)
        # at this moment, negative update is also killed
        if mueff == 0:
            return

        # Update sigma
        if self.ssa == 'CSA':
            self.ps_factor = (1 - self.cs) ** 2 * self.ps_factor + self.cs * (2 - self.cs)
            self.ps *= (1 - self.cs)
            self.ps += math.sqrt(self.cs * (2 - self.cs) * mueff) * self.sigma * self.transform_inverse(np.dot(w[w > 0], sary[w > 0]))
            normsquared = np.sum(self.ps * self.ps)
            self.sigma *= math.exp((math.sqrt(normsquared) / self.chiN - math.sqrt(self.ps_factor)) * self.cs / self.ds)
            # hsig = normsquared < (2.0 + 4.0 / (self.N + 1)) * self.N
            hsig = normsquared < ((1.4 + 2.0 / (self.N + 1)) * self.chiN * math.sqrt(self.ps_factor)) ** 2
        elif self.ssa == 'TPA':
            # find symmetric points
            nsary = sary / np.linalg.norm(sary, axis=1)[np.newaxis].T
            ndx = self.dx / np.linalg.norm(self.dx)
            for ip in range(lam):
                if np.allclose(nsary[ip], ndx):
                    break
            for im in range(lam):
                if np.allclose(nsary[im], -ndx):
                    break
            # update
            if ip < lam and im < lam:
                alpha_act = im - ip
                alpha_act /= float(lam - 1)
                self.ps += self.cs * (alpha_act - self.ps)
                self.sigma *= math.exp(self.ps / self.ds)
                hsig = self.ps < 0.5
            else:
                hsig = True
        elif self.ssa == 'HSA':
            # Akimoto. Ph.D. Thesis.
            self.ps *= (1 - self.cs)
            self.ps += math.sqrt(self.cs * (2 - self.cs) * mueff) * self.sigma * self.transform_inverse(np.dot(w[w > 0], sary[w > 0]))
            musig = np.dot(w[w > 0], np.linalg.norm(self.transform_inverse(sary[w > 0]), axis=1)) * self.sigma
            psnorm = np.linalg.norm(self.ps)
            self.sigma *= math.exp((((1 - self.asig) * musig + self.asig * psnorm) / self.chiN - 1) / self.ds)
            hsig = psnorm**2 < (2 + 4 / (self.N + 1)) * self.N
        elif self.ssa == 'NCSA':
            nstep = np.dot(w[w > 0], self.transform_inverse(sary[w > 0])) * self.sigma
            self.ps *= (1 - self.cs)
            self.ps += math.sqrt(self.cs * (2 - self.cs) / np.sum(nstep * nstep)) * nstep
            normsquared = np.sum(self.ps * self.ps)
            self.sigma *= math.exp((normsquared - 1) * self.cs / self.ds)
            hsig = normsquared < 2.0
        elif self.ssa == 'MICSA':
            sigfac, hsig = self.mixed_integer_handler.mixed_integer_csa(w, self.sigma * self.transform_inverse(sary), self.coordinate_std)
            self.sigma *= sigfac
        elif self.ssa == 'NoSSA':
            hsig = 1
        else:
            raise NotImplementedError('ssa = ' + self.ssa + ' is not implemented.')

        # Update xmean
        self.dx = np.dot(w[w > 0], sarx[w > 0]) - np.sum(w[w > 0]) * self.xmean
        self.xmean += self.cm * self.dx

        # Cumulation
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * mueff) * np.dot(w[w > 0], sary[w > 0])
        self.pc_factor = (1 - self.cc) ** 2 * self.pc_factor + hsig * self.cc * (2 - self.cc)

        # Rank-mu
        if self.cmu == 0:
            rank_mu = np.zeros((self.N, self.N))
        elif self.opts.active_update:
            rank_mu = np.dot(sary.T * w, sary) - np.sum(w) * self.cov
        else:
            rank_mu = np.dot(sary[w > 0].T * w[w > 0], sary[w > 0]) - np.sum(w[w > 0]) * self.cov
        # Rank-one
        if self.cone == 0:
            rank_one = np.zeros((self.N, self.N))
        else:
            rank_one = np.outer(self.pc, self.pc) - self.pc_factor * self.cov
        self.cov += self.cmu * rank_mu + self.cone * rank_one
        # update the square root of the covariance matrix
        if (self.neval - self.eigneval) > self.opts.eigen_frequency:
            self._decompose(det_normalize=self.opts.det_normalization)

        # Inject Two Symmetric Solutions for TPA
        if self.ssa == 'TPA':
            tmp = self.transform_inverse(self.dx)
            dy = (np.linalg.norm(np.random.randn(self.N)) /
                  np.sqrt(np.dot(tmp, tmp))) * self.dx
            inj = self.cls_solution(self.xmean + dy)
            inj.arxmut[:] = 0.0
            self.inject_solution([inj])
            inj = self.cls_solution(self.xmean - dy)
            inj.arxmut[:] = 0.0
            self.inject_solution([inj])

    def __init__(self, opts, func):
        super(Cma, self).__init__(opts)

        self.N = self.opts.N
        self.chiN = np.sqrt(self.N) * (1.0 - 1.0 / (4.0 * self.N) + 1.0 / (21.0 * self.N * self.N))
        self.func = func
        self._xmean = np.array(self.opts.xmean)
        if isinstance(self.opts.sigma, np.ndarray):
            self.sigma = np.exp(np.log(self.opts.sigma).mean())
            self.D = self.opts.sigma / self.sigma
        else:
            self.sigma = self.opts.sigma
            self.D = np.ones(self.N)
        self.cov = np.diag(self.D**2)  # cov = B * D^2 * B^t
        self.pc_factor = 0.0
        self.pc = np.zeros(self.N)
        self.dx = np.zeros(self.N) * np.nan
        self.sigma *= self._decompose(det_normalize=True)
        self.popsize = self.opts.lam

    def expected_update_snorm(self):
        """Compute the expected square norm of the normalized parameter update

        Returns
        -------
        float
            expected square norm of the normalized parameter update
        """
        N = self.N
        w = self.w
        if self.ssa == 'CSA':
            sfactor_a6 = 4. * self.ps_factor * (N / self.chiN ** 2 - 1) * (self.cs / self.ds) ** 2
            sfactor_a5 = 1 + sfactor_a6 * 2
        elif self.ssa == 'NoSSA':
            sfactor_a5 = 1.0
            sfactor_a6 = 0.0
        else:
            raise NotImplementedError()
        if self.opts.active_update:
            v = w
        else:
            v = np.zeros(w.shape)
            v[w > 0] = w[w > 0]
        mfactor = self.cm**2 * N * np.sum(w[w > 0] ** 2)
        rankmu_factor = (N * (N + 1)) * (self.cmu**2 * np.dot(v, v))
        muone_factor = (N * (N + 1)) * (self.cc * (2.0 - self.cc) * self.cone * self.cmu * np.sum(w[w > 0]**2 * v[w > 0]) / np.sum(w[w > 0] ** 2))
        rankone_factor = self.cone**2 * (self.pc_factor**2 * N**2 + (2 * self.pc_factor**2 - 2 * self.pc_factor + 1) * N)
        cfactor = rankmu_factor + muone_factor + rankone_factor
        return mfactor + (sfactor_a5 * cfactor + N * sfactor_a6) / 2


class CmaOption(ABCCmaOption):
    def __init__(self,
                 N = '0',
                 sigma = '0.0',
                 xmean = 'np.zeros(N)',
                 lam = '4 + int(3 * np.log(N))',
                 active_update = 'False',
                 w = 'np.log((lam + 1) / 2.0) - np.log(np.arange(1, lam+1))',
                 mueff = 'np.sum(w[w>0])**2 / np.sum(w[w>0] ** 2)',
                 cm = '1 # learning rate for m-update',
                 cc='(4+mueff/N) / (N+4+2*mueff/N)',
                 cone='2 / ((N+1.3)**2+mueff)',
                 cmu='min(1-cone, 2 * (mueff-2+1/mueff) / ((N+2)**2+mueff))',
                 eigen_frequency='lam/(10*N*(cone+cmu)) # decomposition performed after given #f-calls',
                 ssa='"CSA" # ("CSA", "TPA", "NCSA", "HSA", "MICSA", "NoSSA")',
                 tpa_cs='0.3',
                 tpa_ds='np.sqrt(N)',
                 csa_cs='(mueff+2)/(N+mueff+5)',
                 csa_ds='1 + 2.0*max(0, np.sqrt((mueff-1)/(N+1))-1) + csa_cs',
                 ncsa_cs='3.0 / (6.0 + N/mueff**0.9)',
                 ncsa_ds='2.0',
                 hsa_cs='(mueff+2)/(N+mueff+5)',
                 hsa_ds='(1 + hsa_cs) / (1 + hsa_cs - hsa_as)',
                 hsa_as='np.sqrt((N + 1) / (N + mueff))',
                 det_normalization='False # (bool)',
                 **kwargs):
        super(CmaOption, self).__init__(**kwargs)
        self.setattr_from_local(locals())


class PsaCma(Cma):
    pass


class PsaCmaOption(CmaOption):
    def __init__(self,
                 # Initial Parameters of the Sampling Distribution
                 N='0',
                 sigma='0.0',
                 xmean='np.zeros(N)',
                 # Parameters for the Population Size Adaptaiton
                 psa='True # (bool) Population Size Adaptation Mode',
                 psa_lam_min='int(lam) # minimum population size',
                 psa_lam_max='np.inf # maximum population size',
                 psa_alpha='1.4 # threshold',
                 psa_beta='0.4 # cumulation factor for normalized update',
                 psa_dt='1/float(psa_beta) # dampling for popsize update',
                 psa_granularity='1 # guranularity of adapted popsize',
                 **kwargs):
        super(PsaCmaOption, self).__init__(**kwargs)
        self.setattr_from_local(locals())


if __name__ == '__main__':
    ## Requirements for the objective function
    # argument `x`: a single candidate solution or a matrix of candidate solutions. np.ndarray (`x.ndim in (1, 2)`)
    # - If `x.ndim == 2`, then `x.shape` is in the order (population size, objective function dimensionality).
    # - For parallel simulation, it is necessary to accept `x` such that `x.ndim == 2`. (Parallelization must be implemented separately in the function.)
    # The return value is an evaluation value or a one-dimensional array of evaluation values for each solution.
    def sphere(x):
        x = np.array(x)
        if x.ndim == 1:
            return np.sum(x**2)
        elif x.ndim == 2:
            return np.sum(x**2, axis=1)
        else:
            raise ValueError('x.ndim has to be <= 2, but given x.ndim was {}'.format(x.ndim))

    def rastrigin(x):
        a = 10.0
        x = np.array(x)
        if x.ndim == 1:
            return np.sum(x**2 + a * (1 - np.cos(2 * np.pi * x)))
        elif x.ndim == 2:
            return np.sum(x**2 + a * (1 - np.cos(2 * np.pi * x)), axis=1)
        else:
            raise ValueError('x.ndim has to be <= 2, but given x.ndim was {}'.format(x.ndim))
        
    ## FuncWrapper
    # Pass a function object to `func`.
    # If multiple solutions can be evaluated in parallel, set `batcheval` to `True`.
    # Otherwise (default), `batcheval=False`.
    # fobj = FuncWrapper(func=sphere, batcheval=True)
    fobj = FuncWrapper(func=rastrigin, batcheval=True)

    ## Configuration object generation
    ### Minimum parameters that must be entered.
    # - `N`: Number of dimensions of the objective function.
    # - `sigma`: Initial step size.
    # - `xmean`: Initial mean vector
    ### Parameters that can be adjusted as needed
    # - `psa_lam_min`: minimum population size, natural number, minimum `4`. Default `4 + floor(3*log(N))`.
    # - `psa_lam_max`: maximum population size, natural number, minimum `4`. default `np.inf`.
    # - `psa_granularity`: granularity of the population size, natural number, minimum `1`. Default `1`. The conversion of a real-valued population size to a natural number is made to a multiple of the value specified here.
    # - `parallel`: `True` if the solutions are evaluated in parallel. Default `False`.
    # - The solution population is evaluated in parallel only if `batcheval=True` in FuncWrapper and this `parallel` is set to `True`. In this case, the input to the objective function is a two-dimensional array.
    # - Note that an error occurs if `batcheval=False` and `parallel=True`.
    # - In other cases, the functions are evaluated one by one. In this case, the input to the destination function is a 1D array.
    ### Exit Conditions
    # The terminating condition begins with `check_`.
    # By default, the search continues until the distribution converges and no significant mutations are found on the computer.
    # Example.)
    # - `check_target`: The target for the best solution evaluation.
    # - `check_maxruntime`: The maximum number of function evaluations.
    # - `check_maxiter`: the maximum number of iterations. `check_maxiter`: The maximum number of updates of the distribution.
    # - `check_maxsec`: The maximum execution time in seconds.
    ### Logging parameters
    # The parameters starting with `log_`.
    # - `log_display`: Display the progress (number of iterations, number of evaluations, elapsed time, and the value of the best solution obtained for the iteration) at runtime.
    # - `log_prefix`: Prefix of the log file. The default is `"'. `log_prefix`, so that the log file is created as a hidden file.
    # - `log_span`: The logging interval, which can be 0, a positive real number less than 1, or an integer greater than or equal to 1.
    #     - `0`: no logging.
    #     - `<1`: interval at which the ratio of the logging time to the algorithm computation time is around the specified value.
    #     - Integer: Log when the number of elapsed iterations is a multiple of the specified value.
    # - `log_variable_list`: The list of parameters to log. If you enter an empty list, only the best solution values per iteration will be logged.
    #    Example.)
    #    - `xmean`: Mean vector of the distribution
    #    - `sigma`: Step size
    #    - `D`: Square root of the eigenvalues of the covariance matrix of the distribution
    #    - `popsize`: the population size (integer).
    #    - `popsize_adaptation.lam_dash`: population size (real number) kept inside PSA    
    opts = PsaCmaOption(N=10, sigma=2, xmean='sigma*np.ones(N)')
    opts.psa_granularity = 10
    opts.parallel = True
    opts.check_target = 1e-8
    opts.check_maxruntime = 1e5
    opts.log_display = True
    opts.log_prefix = repr('test')
    opts.log_span = 0.1
    opts.log_variable_list = repr(['xmean', 'D', 'sigma', 'popsize', 'popsize_adaptation.lam_dash'])

    ## Run
    algo = PsaCma(opts=opts, func=fobj)
    algo.run()

    ## plot
    # The `plot` method allows you to generate a figure from a log file.
    # - `prefix`: The log file prefix.
    # - `variable_list`: The list of parameters to plot.
    # - `xaxis`: value for the horizontal axis.
    #     - `0`: number of iterations.
    #     - `1`: number of evaluations.
    #     - `2`: execution time.
    # It is possible to omit the `prefix` and `variable_list` by using the `plotme` method.
    # It returns a `figure` object (first return value) for the entire figure and a `dictionary` (second return value) of `axis` objects for each parameter.
    # - The figure can be modified after it has been generated, as in the following example. (The display will reflect this only when in `nbagg` mode.)
    fig, axdict = algo.plot(prefix=algo.opts.log_prefix, variable_list=algo.opts.log_variable_list, xaxis=0)
    #fig, axdict =  algo.plotme()
    for key in axdict: 
        if all(line.get_ydata().min() > 0 for line in axdict[key].get_lines()):
            axdict[key].set_yscale('log')
    # plt.savefig('test.eps')
    plt.savefig('test.png')