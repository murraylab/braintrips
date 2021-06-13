from .params.hemodynamic import param_dict
from .params.synaptic import sigma
from .utils import cov_to_corr, perform_gsr
from .utils import solve_lyapunov_fast as solve_lyapunov

from scipy.sparse import csc_matrix
import numpy as np

# dtype = np.float32  # SPEED
dtype = np.float64  # PRECISION


class Balloon:
    """ Class for hemodynamic transfer function for input-state-output."""

    def __init__(self, nareas, syn_noise_var=sigma, linearize=False):
        """
        Parameters
        ----------
        nareas : int
            Number of areas (nodes) in the model
        syn_noise_var : float, optional
            Synaptic noise variance. By default, value of parameter `sigma`
            defined in lib.params.synaptic
        linearize : boolean, optional
            If True, linearized hemodynamic equations are used
        """

        self.gbc_cov = None

        # Number of cortical areas
        self._nareas = nareas

        # Synaptic noise variance
        self._sigma = syn_noise_var

        # Load paramter values
        params = param_dict()
        keys = ['V0', 'kappa', 'gamma', 'tau', 'alpha', 'rho', 'k1', 'k2', 'k3']
        self.V0 = self.kappa = self.gamma = self.tau = self.alpha = None
        self.rho = self.k1 = self.k2 = self.k3 = None
        assert np.all(np.in1d(keys, list(params.keys())))
        for key in keys:
            setattr(self, key, params[key])

        # Steady-state values
        self._x0 = 0.0
        self._f0 = 1.0
        self._v0 = 1.0
        self._q0 = 1.0
        self._y0 = 0.0

        # Hemodynamic state variables
        self._x = self._f = self._v = self._q = self._y = None
        self.reset_state()

        self._linearize = False
        if linearize:
            self._linearize = True
            self._dx = self.xdotlin
            self._df = self.fdotlin
            self._dv = self.vdotlin
            self._dq = self.qdotlin
            self._compute_y = self.ylin
        else:
            self._dx = self.xdot
            self._df = self.fdot
            self._dv = self.vdot
            self._dq = self.qdot
            self._compute_y = self.ynonlin

        self._A = self._Q = self._B = self.Afull = None
        self._cov = self._corr = None
        self.build_jacobian()

    def reset_state(self):
        """ Reset state variables to steady-state values. Note that
        hemodynamic variables f, v, and q are all normalized w.r.t. their
        value at rest, while x and y are in dimensionless units representing
        the percent change from its resting value. """
        self._x = np.repeat(self._x0, self._nareas)  # vasodilatory signal
        self._f = np.repeat(self._f0, self._nareas)  # inflow rate
        self._v = np.repeat(self._v0, self._nareas)  # blood volume
        self._q = np.repeat(self._q0, self._nareas)  # deoxyhemoglobin content
        self._y = np.repeat(self._y0, self._nareas)  # BOLD signal (%)

    def BOLD_tf(self, freqs):
        """
        The analytic solution to the transfer function
        of the BOLD signal y as a function of the input
        synaptic signal z, at a given frequency f, for the
        Balloon-Windkessel hemodynamic model. For derivation
        details see Robinson et al., 2006, BOLD responses to
        stimuli. """
        w = 2 * np.pi * freqs
        beta = (self.rho + (1. - self.rho) * np.log(1. - self.rho)) / self.rho
        T_yz = (self.V0 * (self.alpha * (self.k2 + self.k3) * (
                    w * self.tau * 1.j - 1) + (self.k1 + self.k2) * (
                    self.alpha + beta - 1. - w *
                    self.alpha * beta * self.tau * 1.j))) / (
                    (1. - w * self.tau * 1.j) *
                    (1. - self.alpha * w * self.tau * 1.j) * (
                     w * w + w * self.kappa * 1.j - self.gamma))
        return T_yz * np.conj(T_yz)

    # ----------------------------------------------
    # --- Nonlinear hemodynamic equations ----------
    # ----------------------------------------------

    def xdot(self, z, x, f):
        return z - self.kappa * x - self.gamma * (f - 1.)

    @staticmethod
    def fdot(x):
        return x

    def vdot(self, f, v):
        num = f - np.power(v, 1. / self.alpha)
        if isinstance(num, complex):
            print(f, v, num)
            raise ValueError("Complex value encountered.")
        return num / self.tau

    def qdot(self, f, v, q):
        return (f * (1. - np.power(1. - self.rho, 1. / f)) / self.rho - q * (
            np.power(v, 1. / self.alpha - 1.))) / self.tau

    def ynonlin(self, v, q):
        return self.V0 * (self.k1 * (1. - q) + self.k2 * (
                1. - q / v) + self.k3 * (1. - v))

    # ----------------------------------------------
    # --- Linearized hemodynamic equations ---------
    # ----------------------------------------------

    def xdotlin(self, z, x, f):
        return self.xdot(z, x, f)

    def fdotlin(self, x):
        return self.fdot(x)

    def vdotlin(self, f, v):
        return ((f - 1.) - (v - 1.) / self.alpha) / self.tau

    def qdotlin(self, f, v, q):
        x1 = (1. + (1. / self.rho) * (1. - self.rho) * np.log(
            1. - self.rho)) * (f - 1.)
        x2 = -(q - 1.)
        x3 = (self.alpha - 1) * (v - 1.) / self.alpha
        return (x1 + x2 + x3) / self.tau

    def ylin(self, v, q):
        return self.V0 * ((self.k1 + self.k2) * (
                1. - q) + (self.k3 - self.k2) * (1. - v))

    # ----------------------------------------------
    # --- Methods for BOLD FC linearization --------
    # ----------------------------------------------

    def build_jacobian(self):

        idmat = np.eye(self._nareas)
        zeromat = np.zeros(shape=(self._nareas, self._nareas))

        # ---- Define full (synaptic + hemodynamic) Jacobian
        # and noise covariance matrices --------------------

        # Partials of x w.r.t. hemodynamic quantities x, f, v, q
        A_xx = -self.kappa * idmat
        A_xf = -self.gamma * idmat
        A_xv = zeromat
        A_xq = zeromat

        # Partials of f w.r.t. hemodynamic quantities x, f, v, q
        A_fx = idmat
        A_ff = zeromat
        A_fv = zeromat
        A_fq = zeromat

        # Partials of v w.r.t. hemodynamic quantities x, f, v, q
        A_vx = zeromat
        A_vf = idmat / self.tau
        A_vv = -idmat / (self.alpha * self.tau)
        A_vq = zeromat

        # Partials of q w.r.t. hemodynamic quantities x, f, v, q
        A_qx = zeromat
        A_qf = idmat / self.tau * (1.+(1.-self.rho)*np.log(
            1.-self.rho) / self.rho)
        A_qv = idmat * (self.alpha - 1.) / (self.tau * self.alpha)
        A_qq = idmat * -1. / self.tau

        # Construct hemodynamic sub-block
        hemo_row1 = np.hstack((A_xx, A_xf, A_xv, A_xq))
        hemo_row2 = np.hstack((A_fx, A_ff, A_fv, A_fq))
        hemo_row3 = np.hstack((A_vx, A_vf, A_vv, A_vq))
        hemo_row4 = np.hstack((A_qx, A_qf, A_qv, A_qq))
        hemo_block = np.vstack((hemo_row1, hemo_row2, hemo_row3, hemo_row4))

        # Dependence of hemodynamic state on vasodilatory signal enters
        # through equation for dx/dt, assumed through S_E only
        dState_dSE = np.vstack((idmat, np.zeros((
            3 * self._nareas, self._nareas))))
        dState_dS = np.hstack((dState_dSE, np.zeros(
            (4 * self._nareas, self._nareas))))

        # Construct full 6x6 Jacobian with (S_E, S_I, x, f, v, q)
        A_syn = np.zeros((2 * self._nareas, 2 * self._nareas))
        Q_syn = np.identity(2 * self._nareas) * self._sigma * self._sigma

        input_row = np.hstack((
            A_syn, np.zeros((2 * self._nareas, 4 * self._nareas))))
        state_row = np.hstack((dState_dS, hemo_block))

        # Full Jacobian and noise covariance matrices
        A = np.vstack((input_row, state_row)).astype(dtype)
        self._A = np.asfortranarray(A)

        Q = np.pad(Q_syn, pad_width=(0, 4 * self._nareas),
                   mode='constant', constant_values=0).astype(dtype)
        self._Q = csc_matrix(Q)

        # Dependence of BOLD output signal on state variables
        dydv = idmat * (self.k2 - self.k3) * self.V0
        dydq = -idmat * (self.k1 + self.k2) * self.V0
        self._B = np.hstack(
            (np.zeros((self._nareas, self._nareas * 4)), dydv, dydq)).astype(
            dtype)

    def build_full_jacobian(self, A_syn):

        # idmat = np.eye(self._nareas)
        # zeromat = np.zeros(shape=(self._nareas, self._nareas))
        #
        # # ---- Define full (synaptic + hemodynamic) Jacobian
        # # and noise covariance matrices --------------------
        #
        # # Partials of x w.r.t. hemodynamic quantities x, f, v, q
        # A_xx = -self.kappa * idmat
        # A_xf = -self.gamma * idmat
        # A_xv = zeromat
        # A_xq = zeromat
        # A_xy = zeromat
        #
        # # Partials of f w.r.t. hemodynamic quantities x, f, v, q
        # A_fx = idmat
        # A_ff = zeromat
        # A_fv = zeromat
        # A_fq = zeromat
        # A_fy = zeromat
        #
        # # Partials of v w.r.t. hemodynamic quantities x, f, v, q
        # A_vx = zeromat
        # A_vf = idmat / self.tau
        # A_vv = -idmat / (self.alpha * self.tau)
        # A_vq = zeromat
        # A_vy = zeromat
        #
        # # Partials of q w.r.t. hemodynamic quantities x, f, v, q
        # A_qx = zeromat
        # A_qf = idmat / self.tau * (1.+(1.-self.rho)*np.log(
        #     1.-self.rho) / self.rho)
        # A_qv = idmat * (self.alpha - 1.) / (self.tau * self.alpha)
        # A_qq = idmat * -1. / self.tau
        # A_qy = zeromat
        #
        # # Partials of y w.r.t. hemodynamic quantities x, f, v, q
        # cv = idmat * self.V0 * (self.k2 - self.k3)
        # cq = - idmat * self.V0 * (self.k1 + self.k2)
        # A_yx = zeromat
        # A_yf = cv / self.tau + cq / self.tau * (
        #         1.+(1.-self.rho)*np.log(1.-self.rho) / self.rho)
        # A_yv = -cv / (self.alpha * self.tau) + cq * (self.alpha-1) / (
        #         self.alpha * self.tau)
        # A_yq = -cq / self.tau
        # A_yy = zeromat
        #
        # # Construct hemodynamic sub-block
        # hemo_row1 = np.hstack((A_xx, A_xf, A_xv, A_xq, A_xy))
        # hemo_row2 = np.hstack((A_fx, A_ff, A_fv, A_fq, A_fy))
        # hemo_row3 = np.hstack((A_vx, A_vf, A_vv, A_vq, A_vy))
        # hemo_row4 = np.hstack((A_qx, A_qf, A_qv, A_qq, A_qy))
        # hemo_row5 = np.hstack((A_yx, A_yf, A_yv, A_yq, A_yy))
        # hemo_block = np.vstack((
        #     hemo_row1, hemo_row2, hemo_row3, hemo_row4, hemo_row5))
        #
        # # Dependence of hemodynamic state on vasodilatory signal enters
        # # through equation for dx/dt, assumed through S_E only
        # dState_dSE = np.vstack((idmat, np.zeros((
        #     4 * self._nareas, self._nareas))))
        # dState_dS = np.hstack((dState_dSE, np.zeros(
        #     (5 * self._nareas, self._nareas))))
        #
        # # Construct full 6x6 Jacobian with (S_E, S_I, x, f, v, q)
        #
        # input_row = np.hstack((
        #     A_syn, np.zeros((2 * self._nareas, 5 * self._nareas))))
        # state_row = np.hstack((dState_dS, hemo_block))
        #
        # # Full Jacobian and noise covariance matrices
        # A = np.vstack((input_row, state_row)).astype(dtype)
        # self.Afull = A

        # Update synaptic sub-block of full Jacobian matrix
        self.update_A_syn(A_syn)
        bold_A = np.array(self._B.dot(self._A.dot(self._B.conj().T)))
        return bold_A

    def update_A_syn(self, A_syn):
        """
        Update synaptic sub-block of full Jacobian matrix.

        Parameters
        ----------
        A_syn : np.ndarray
            synaptic Jacobian matrix (shape (2N, 2N))

        """
        self._A[:2 * self._nareas, :2 * self._nareas] = A_syn
        assert self._A.dtype == dtype

    def moments_method(self, A_syn, gsr=False):
        """
        Solve for the linearized covariance matrix and compute the analytic FC.

        Parameters
        ----------
        A_syn : np.ndarray
            Synaptic Jacobian matrix with shape (2N, 2N), where N is the number
            of areas (i.e., nodes) and the first (last) N rows/columns
            correspond to the excitatory (inhibitory) synaptic gating variables
        gsr : bool, optional
            perform GSR on the covariance matrix prior to computing correlation

        Notes
        -----
        The `cov` and `corr` class attributes are updated by this class. Using
        builtin=True is NOT RECOMMENDED and will run ~100x slower.

        """

        # Update synaptic sub-block of full Jacobian matrix
        self.update_A_syn(A_syn)

        hemo_cov = solve_lyapunov(self._A, self._Q.todense())
        bold_cov = np.array(self._B.dot(hemo_cov.dot(self._B.conj().T)))
        self.gbc_cov = bold_cov.sum(axis=0)
        # else:
        #     bold_cov = solve_lyapunov2(self._A, self._Q, self._B)
        if gsr:
            x = perform_gsr(bold_cov)
        else:
            x = bold_cov
        self._cov = x
        self._corr = cov_to_corr(x)

    # ----------------------------------------------
    # --- Methods for numerical simulation ---------
    # ----------------------------------------------

    def step(self, dt, z):
        """
        Evolve hemodynamic equations forward in time by `dt`, updating state
        variables x, f, v, q, and y.

        Parameters
        ----------
        dt : float
            Differentiation time step
        z : np.ndarray
            Excitatory synaptic gating variables with shape (1, nareas)
        """
        assert z.ndim == 1 and z.size == self._nareas

        # Compute change in each hemodynamic variable using current state and
        # external input `z`
        dx = self._dx(z, self._x, self._f) * dt
        df = self._df(self._x) * dt
        dv = self._dv(self._f, self._v) * dt
        dq = self._dq(self._f, self._v, self._q) * dt

        # Update hemodynamic state variables
        self._x += dx
        self._f += df
        self._v += dv
        self._q += dq

        # Using updated state variables, compute new BOLD signal
        self._y = self._compute_y(self._v, self._q)

    # ------------------------
    # --- Properties ---------
    # ------------------------

    @property
    def state(self):
        """
        Hemodynamic state variables.

        Returns
        -------
        5*N rows by N columns, where N is the number of nodes. Rows correspond
        to hemodynamic variables x, v, f, q, and y.
        """
        return np.vstack((self._x, self._f, self._v, self._q, self._y))

    @property
    def B(self):
        """
        Partial derivatives of the BOLD signal with respect to the two synaptic
        gating variables and the six hemodynamic state variables.

        Returns
        -------
        np.ndarray
            shape (N, 6*N), where N is the number of nodes

        """
        return self._B

    @property
    def x(self):
        """
        Vasodilatory signal.

        Returns
        -------
        np.ndarray

        """
        return self._x

    @property
    def f(self):
        """
        Blood inflow rate (normalized w.r.t. resting-state value).

        Returns
        -------
        np.ndarray

        """
        return self._f

    @property
    def v(self):
        """
        Blood volume (normalized w.r.t. resting-state value).

        Returns
        -------
        np.ndarray

        """
        return self._v

    @property
    def q(self):
        """
        Deoxyhemoglobin content (normalized w.r.t. resting-state value).

        Returns
        -------
        np.ndarray
        """
        return self._q

    @property
    def y(self):
        """
        BOLD signal (% change relative to resting-state value).

        Returns
        -------
        np.ndarray

        """
        return self._y

    @property
    def cov(self):
        """
        BOLD covariance matrix (approximated analytically).

        Returns
        -------
        np.ndarray
            Covariance matrix of linearized fluctuations about the
            fixed point.
        """
        return self._cov

    @property
    def var(self):
        """
        BOLD variances.

        Returns
        -------
        np.ndarray
        """
        return np.diag(self._cov)

    @property
    def corr(self):
        """
        BOLD FC matrix (approximated analytically).

        Returns
        -------
        np.ndarray
            FC matrix of linearized fluctuations about the fixed point.
        """
        return self._corr
