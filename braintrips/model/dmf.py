""" Dynamic mean field model base class. """

from .utils import cov_to_corr, load_model_params
from .utils import perform_gsr
from .sim import Sim
from .hemo import Balloon

import numpy as np
from scipy.linalg import eig, inv, eigvals
from scipy.linalg import solve_lyapunov
from scipy.optimize import fsolve


class Model:
    """ Mean-field model of neural circuit dynamics. """

    def __init__(self, sc, g=0, h_map=None, w_ee=(0.15, 0.),
                 w_ie=(0.15, 0.), syn_params=None, htr2a_map=None, de=0,
                 di=0, verbose=True, incr_hmap=True):
        """

        Parameters
        ----------
        sc : ndarray
            Structural connectivity matrix
        g : float, optional (default 0)
            Global coupling parameter to scale structural connections. If zero,
            network consists of dynamically uncoupled nodes.
        h_map : ndarray, optional
            Heterogeneity map with which to parametrize recurrent excitation
            strength. If None, the nodes all have equivalent local dynamical
            properties.
        w_ee : tuple, optional (default (0.15,0))
            Local recurrent excitatory connectivity weights w^{EE}.
            Requires a tuple with size 2 as ( w_{min}, w_{scale} ).
        w_ie : tuple, optional (default (0.15,0))
            Local excitatory to inhibitory connectivity weights (w^{EI}). 
            Requires a tuple with size 2 as ( w_{min}, w_{scale} ).
        syn_params : dict, optional
            Synaptic dynamical model parameters
        htr2a_map : ndarray
            HTR2A gene expression map
        de : float
            fractional change in gain on excitatory neural ensembles
        di : float
            fractional change in gain on inhibitory neural ensembles
        verbose : bool, optional
            If True, prints diagnostics to console (True by default)  
        
        """

        # Print diagnostics to console
        self._verbose = verbose

        # Global coupling
        self._G = g

        # Structural connectivity
        self._sc = np.copy(sc)
        self._nareas = self._sc.shape[0]
        self._id = np.identity(self._nareas)

        # For simulation results
        self._sim = None

        # If custom model parameters were provided, load model parameters
        # with corresponding keys
        model_params = load_model_params()
        if syn_params is not None:
            for key in syn_params.keys():
                model_params[key] = syn_params[key]

        # Dynamical instability flag
        self._is_unstable = False

        # Synaptic (i.e., gating variables) Jacobian matrices
        self._A = None
        self._cov = self._corr = self._cov_bold = self._corr_bold = None

        # Set model parameters
        self._w_II = np.repeat(model_params['w_II'], self._nareas)
        self._w_IE = np.repeat(model_params['w_IE'], self._nareas)
        self._w_EE = np.repeat(model_params['w_EE'], self._nareas)
        self._w_EI = np.repeat(model_params['w_EI'], self._nareas)
        self._I0 = np.repeat(model_params['I0'], self._nareas)
        self._J_NMDA = np.repeat(model_params['J_NMDA'], self._nareas)
        self._sigma = model_params['sigma']
        self._gamma_kinetic = model_params['gamma']
        self._W_I = model_params['W_I']
        self._W_E = model_params['W_E']
        self._tau_I = model_params['tau_I']
        self._tau_E = model_params['tau_E']
        self._d_I = model_params['d_I']
        self._d_E = model_params['d_E']
        self._b_I = model_params['b_I']
        self._b_E = model_params['b_E']
        self._a_I = model_params['a_I']
        self._a_E = model_params['a_E']
        self._I_ext = np.repeat(model_params['I_ext'], self._nareas)

        # Baseline input currents to each population
        self._I0_E = self._W_E * self._I0
        self._I0_I = self._W_I * self._I0

        # Steady state values (for isolated node)
        self._I_E_ss = np.repeat(model_params['I_E_ss'], self._nareas)
        self._I_I_ss = np.repeat(model_params['I_I_ss'], self._nareas)
        self._S_E_ss = np.repeat(model_params['S_E_ss'], self._nareas)
        self._S_I_ss = np.repeat(model_params['S_I_ss'], self._nareas)
        self._r_E_ss = np.repeat(model_params['r_E_ss'], self._nareas)
        self._r_I_ss = np.repeat(model_params['r_I_ss'], self._nareas)

        # Noise covariance matrix
        self._Q = np.identity(2 * self._nareas) * self._sigma * self._sigma

        # Balloon-Windkessel model, here using the "delegation" programming
        # pattern in which responsibility for implementing the synaptic-to-bold
        # operation is delegated to a different object
        self._bw = Balloon(
            nareas=self._nareas, syn_noise_var=self._sigma, linearize=False)

        # Compute baseline (unmodulated) transfer function and its derivatives
        self._set_transfer_funcs(baseline=True)

        # Heterogeneity map values
        self._raw_hmap = h_map
        self._hmap = 0.0

        # Recurrent excitation heterogeneity
        if self._raw_hmap is not None:
            hmap_range = np.ptp(self._raw_hmap)
            self._hmap = -(self._raw_hmap - np.max(self._raw_hmap)) / hmap_range
            if not incr_hmap:
                self._hmap *= -1
            assert np.all(self._hmap <= 1)
            assert np.all(self._hmap >= 0)
            self._w_EE = self.affine(x=self._hmap, m=w_ee[1], b=w_ee[0])
            self._w_IE = self.affine(x=self._hmap, m=w_ie[1], b=w_ie[0])

        # Partial derivative of current wrt gating variables
        self._K_EE = (self._w_EE * self._J_NMDA * self._id) + (
                self._G * self._J_NMDA * self._sc)
        self._K_IE = self._w_IE * self._J_NMDA * self._id
        self._K_II = -self._w_II * self._id
        self._K_EI = None  # Assigned once FIC is computed

        # Solve for feedback inhibition strength (PRIOR to gain modulation)
        self._compute_FIC()

        # Implement neural gain modulation
        self._di = di
        self._de = de
        if htr2a_map is not None and htr2a_map.size > 1:
            if len(set(htr2a_map)) == 1:
                self._gain_map = htr2a_map
            else:
                htr2a_range = np.ptp(htr2a_map)
                self._gain_map = (htr2a_map - np.min(htr2a_map)) / htr2a_range
                assert np.all(self._gain_map <= 1)
                assert np.all(self._gain_map >= 0)
        elif htr2a_map is not None:
            self._gain_map = htr2a_map
        else:
            self._gain_map = None

        if de and di:
            assert self._gain_map is not None
            self.update_gain_params(de, di)
        else:
            self.__solve_fixed_point()
            self.compute_jacobian()

        # Initialize state variables to their steady-state values
        self._I_E = None
        self._I_I = None
        self._S_E = None
        self._S_I = None
        self._r_E = None
        self._r_I = None
        self.reset_syn_state()

        self.gbc_cov = None

    def _set_transfer_funcs(self, baseline):
        """
        Compute and store transfer functions as class attributes.

        Parameters
        ----------
        baseline : bool
            if False, an HTR2A map must have been provided upon initialization;
            transfer functions will then be re-computed using the values of
            alpha and gamma currently stored in corresponding class attributes.
        """

        if not baseline:
            assert self._gain_map is not None
            ge = self._gain_map * self._de
            gi = self._gain_map * self._di
            self.a_E_eff = self._a_E * (1. + ge)
            self.a_I_eff = self._a_I * (1. + gi)
        else:
            self.a_E_eff = self._a_E
            self.a_I_eff = self._a_I

        def phi_E(I_E):
            x = self.a_E_eff * I_E - self._b_E
            return x / (1. - np.exp(-self._d_E * x))

        def phi_I(I_I):
            x = self.a_I_eff * I_I - self._b_I
            return x / (1. - np.exp(-self._d_I * x))

        def dphi_E(I_E):
            x = -self._d_E * (self.a_E_eff * I_E - self._b_E)
            expx = np.exp(x)
            return self.a_E_eff * (1. - expx + x * expx) / ((1. - expx) ** 2)

        def dphi_I(I_I):
            x = -self._d_I * (self.a_I_eff * I_I - self._b_I)
            expx = np.exp(x)
            return self.a_I_eff * (1. - expx + x * expx) / ((1. - expx) ** 2)

        self.H_E = phi_E
        self.H_I = phi_I
        self.dHE_dIE = dphi_E
        self.dHI_dII = dphi_I

    def _compute_FIC(self):
        """ Analytically solve for feedback inhibition strength in each area."""

        # Numerically solve for steady state inhibitory current
        I_I_ss, infodict, ier, mesg = fsolve(
            self.__func, x0=self._I_I_ss, full_output=True)

        # Update values of steady-state class attributes
        self._I_I_ss = np.copy(I_I_ss)
        self._r_I_ss = self.H_I(self._I_I_ss)
        self._S_I_ss = np.copy(self._r_I_ss) * self._tau_I

        # Solve for J analytically
        w_EI = (self._I_ext - self._I_E_ss + self._W_E * self._I0 +
                self._K_EE.dot(self._S_E_ss)) / self._S_I_ss
        self._w_EI = w_EI
        self._K_EI = -self._w_EI * self._id

    def __func(self, I):
        """ Root of this equation yields the steady-state inhibitory current
        (Auxiliary method intended for use with self.__compute_FIC). """
        return self._I0_I + self._K_IE.dot(self._S_E_ss) - (
                self._w_II * self._tau_I * self.H_I(I)) - I

    def __solve_fixed_point(self):
        """Solve for the new system fixed point, and place the current state of
        the system at that point. """
        steady_state, infodict, ier, mesg = fsolve(
            self.__dS, x0=np.concatenate((self._S_E_ss, self._S_I_ss)),
            full_output=True)
        self._S_E_ss, self._S_I_ss = steady_state.reshape((2, self.nareas))
        self._I_E_ss = self.exc_current(self._S_E_ss, self._S_I_ss)
        self._I_I_ss = self.inh_current(self._S_E_ss, self._S_I_ss)
        self._r_E_ss = self.H_E(self._I_E_ss)
        self._r_I_ss = self.H_I(self._I_I_ss)
        self.reset_syn_state()

    def __dS(self, state):
        """ Root of this function yields the steady-state gating variables
        (Auxiliary method intended for use with self.__solve_fixed_point). """
        S_E = state[:self._nareas]
        S_I = state[self._nareas:]
        I_E = self.exc_current(S_E, S_I)
        I_I = self.inh_current(S_E, S_I)
        r_E = self.H_E(I_E)
        r_I = self.H_I(I_I)
        dSE = -(S_E / self._tau_E) + (self._gamma_kinetic * r_E * (1. - S_E))
        dSI = -(S_I / self._tau_I) + r_I
        return np.concatenate((dSE, dSI))

    def compute_jacobian(self):
        """
        Compute the Jacobian matrix.

        Notes
        -----
        This method must be executed before calculating the linearized
        covariance or performing numerical simulations, AND each time any
        model parameters are changed!!
        """

        # Derivatives of transfer function for each cell type
        # at steady state value of current
        dr_E = self.dHE_dIE(self._I_E_ss) * self._id
        dr_I = self.dHI_dII(self._I_I_ss) * self._id

        # A_{mn} = dS_i^m/dS_j^n
        A_EE = (-1. / self._tau_E - (self._gamma_kinetic * self._r_E_ss)
                ) * self._id + ((-self._gamma_kinetic * (self._S_E_ss - 1.))
                                * self._id).dot(dr_E.dot(self._K_EE))

        A_EI = ((self._gamma_kinetic * (1. - self._S_E_ss)) * self._id).dot(
            dr_E.dot(self._K_EI))
        A_IE = dr_I.dot(self._K_IE)
        A_II = (-1. / self._tau_I) * self._id + dr_I.dot(self._K_II)

        # Stack blocks to form full Jacobian
        col1 = np.vstack((A_EE, A_IE))
        col2 = np.vstack((A_EI, A_II))
        self._A = np.hstack((col1, col2))

        # Eigenvalues of Jacobian matrix
        evals = eigvals(self._A)

        # Check stability using eigenvalues
        self._is_unstable = evals.real.max() >= 0

    def moments_method(self, bold=True, gsr=False):
        """ 
        Solve for the linearized covariance matrix and compute the analytic FC.

        Parameters
        ----------
        bold : bool, optional
            if True, also compute the BOLD signal covariance and FC matrix
        gsr : bool, optional
            perform GSR on the covariance matrix prior to computing correlation

        Notes
        -----
        The `cov` and `corr` class attributes are updated by this class.

        """

        if self._is_unstable:
            print("System is dynamically unstable and covariance is undefined.")
            self._cov = self._corr = None
            return
        elif bold:
            self._bw.moments_method(self._A, gsr=gsr)
            self._cov_bold = self._bw.cov
            self._corr_bold = self._bw.corr
        else:
            cov = solve_lyapunov(self._A, -self._Q)
            cov = cov[:self.nareas, :self.nareas]
            self.gbc_cov = cov.sum(axis=0)
            if gsr:
                cov = perform_gsr(cov)
            corr = cov_to_corr(cov)
            self._cov = cov
            self._corr = corr

    def exc_current(self, S_E, S_I):
        """
        Compute the excitatory current in each node.

        Parameters
        ----------
        S_E : np.ndarray
            Excitatory gating variables
        S_I : np.ndarray
            Inhibitory gating variables

        Returns
        -------
        np.ndarray

        """
        return self._I0_E + self._K_EE.dot(
            S_E) + self._K_EI.dot(S_I) + self._I_ext

    def inh_current(self, S_E, S_I):
        """ 
        Compute the inhibitory current in each node.

        Parameters
        ----------
        S_E : np.ndarray
            Excitatory gating variables
        S_I : np.ndarray
            Inhibitory gating variables

        Returns
        -------
        np.ndarray

        """
        return self._I0_I + self._K_IE.dot(S_E) + self._K_II.dot(S_I)

    def _step(self, dt=1E-4, dI=0):
        """
        Advance the system's state forward in time.

        Parameters
        ----------
        dt : float, optional (default 1E-4)
            Integration time step, in seconds.
        dI : float, optional (default 0)
            excitatory impulse current, in units of nA

        """
        self._I_E = self.exc_current(self._S_E, self._S_I) + dI
        self._I_I = self.inh_current(self._S_E, self._S_I)

        self._r_E = self.H_E(self._I_E)
        self._r_I = self.H_I(self._I_I)

        # Compute change in synaptic gating variables
        dS_I = self._dSIdt() * dt + (
                np.sqrt(dt) * self._sigma * np.random.normal(size=self._nareas))

        dS_E = self._dSEdt() * dt + (
                np.sqrt(dt) * self._sigma * np.random.normal(size=self._nareas))

        # Update class members S_E, S_I
        self._S_E += dS_E
        self._S_I += dS_I

        # Clip synaptic gating fractions
        self._S_E = np.clip(self._S_E, 0., 1.)
        self._S_I = np.clip(self._S_I, 0., 1.)

    def integrate(self, t, dt=1e-4, n_save=10, from_fixed=True, sim_seed=None,
                  impulse=None):
        """Euler integration of the system of stochastic differential equations.

        Parameters
        ----------
        t : int
            Total simulation time in seconds.
        dt : float, optional
            Integration time step in seconds. By default dt is 0.1 msec.
        n_save : int, optional
            Sampling rate (time points). By default n_save is 10, therefore if
            dt is 0.1 msec, all the variables will be sampled at 1 msec.
        from_fixed : boolean, optional
            If True, the simulation will begin using steady state values of the
            parameters, otherwise the last available values will be used (i.e.
            from previous simulations etc.)
        sim_seed : int, optional
            The seed for random number generator.
        impulse : (float, float)
            apply an impulse of excitatory current at simulation start; the
            first & second elements of the tuple correspond to the magnitude,
            in nA, and duration, in sec, of the impulse, respectively

        Notes
        -----
        After simulation, the excitatory synaptic variables' time-series can be
        obtained from Model.sim.S_E.
        """

        self._sim = Sim()

        if sim_seed is None:
            sim_seed = np.random.randint(0, 4294967295)
        np.random.seed(sim_seed)

        # Initialize to fixed point
        if from_fixed:
            self.reset_syn_state()

        # Simulation parameters
        dt_save = dt * n_save
        n_sim_steps = int(t / dt + 1)
        n_save_steps = int(t / dt_save + 1)

        # Synaptic state record
        synaptic_state = np.zeros((6, self._nareas, n_save_steps))
        synaptic_state[:, :, 0] = self.state

        delta = tmax = None
        if impulse is not None:
            delta, tmax = impulse

        if self._verbose:
            print("Beginning simulation.")

        # Main for-loop
        time = 0
        for i in range(1, n_sim_steps):

            dI = 0
            if delta is not None:
                if time < tmax:
                    dI = delta
            # if dI:
            #     print(time, dI)

            self._step(dt, dI)
            time += dt

            # Update state variables
            if not (i % n_save):
                i_save = i // n_save
                synaptic_state[:, :, i_save] = self.state

                if self._verbose and not (i_save % 10000):
                    print("%.1f seconds elapsed..." % (i_save * dt * n_save))

        if self._verbose:
            print("Simulation complete.")

        self._sim.t = t
        self._sim.dt = dt_save
        self._sim.n_save = n_save
        self._sim.t_points = np.linspace(0, t, n_save_steps)
        self._sim.seed = sim_seed

        (self._sim.I_I, self._sim.I_E, self._sim.r_I,
         self._sim.r_E, self._sim.S_I, self._sim.S_E) = synaptic_state

    def _dSEdt(self):
        """
        Time derivative of the excitatory synaptic gating variables (in the
        absense of noise -- i.e., in the steady-state condition).
        
        Returns
        -------
        np.ndarray

        Notes
        -----
        The result is computed using the current state of the system, defined
        by the values of the class attributes for current, rate, and gating
        fractions.
        """
        return -(self._S_E / self._tau_E
                 ) + self._gamma_kinetic * self._r_E * (1. - self._S_E)

    def _dSIdt(self):
        """
        Time derivative of the inhibitory synaptic gating variables (in the
        absense of noise -- i.e., in the steady-state condition).

        Returns
        -------
        np.ndarray

        Notes
        -----
        The result is computed using the current state of the system, defined
        by the values of the class attributes for current, rate, and gating
        fractions.
        """
        return -(self._S_I / self._tau_I) + self._r_I

    def csd(self, freqs, pop='E'):
        """
        Cross-spectral density (CSD) of the synaptic gating variables.

        Parameters
        ----------
        freqs : np.ndarray
            The frequency bins within which to compute the CSD.
        pop : str, optional
            If 'E' ('I'), the CSD of the excitatory (inhibitory) population is
            returned.

        Returns
        -------
        csd : (N,N,M) np.ndarray
            CSD of the specified synaptic population in M frequency bins.

        """
        if self._A is None:
            self.compute_jacobian()

        Id = np.identity(self._nareas * 2)
        power = np.empty(
            (2 * self._nareas, 2 * self._nareas, len(freqs)), dtype=complex)
        sig = complex(self._sigma ** 2)
        for i, f in enumerate(freqs):
            w = 2. * np.pi * f
            M1 = inv(self._A + 1.j * w * Id)
            M2 = inv(self._A.T - 1.j * w * Id)
            M3 = np.dot(M1, M2)
            power[:, :, i] = M3 * sig
        if pop == 'E':
            return power[:self._nareas, :self._nareas, :]
        elif pop == 'I':
            return power[self._nareas:, self._nareas:, :]
        else:
            return power

    def gbc(self, fc):
        """
        Compute GBC map, equivalent to the row-wise average of the input FC
        matrix using off-diagonal entries. FC should first be Fisher
        r-to-Z-transformed.

        Returns
        -------
        (N,) np.ndarray
            BOLD GBC map

        """
        idiag = np.where(np.identity(self._nareas))
        fc[idiag] = 0
        gbc = fc.sum(axis=1) / (self.nareas - 1)
        return gbc

    def update_gain_params(self, de, di, gain_map=None, rebalance_fic=False,
                           bold=False, gsr=False):
        """
        Update parameters and/or used for neural gain parametrization.

        Parameters
        ----------
        de : float
            fractional change in gain on excitatory neural ensembles
        di : float
            fractional change in gain on inhibitory neural ensembles
        gain_map : array_like (optional)
            if argument is passed, use this map to parametrize gain
        rebalance_fic : bool (optional, default False)
            if True, recompute FIC values
        bold: bool, optional
            if True, recompute BOLD FC with moments method
        gsr : bool, optional
            if True (and bold is True), use GSR on BOLD covariance matrix

        """
        if gain_map is not None:  # Update attribute and recompute baseline FIC
            self._gain_map = (gain_map - gain_map.min()) / np.ptp(gain_map)
            assert np.all(self._gain_map <= 1)
            assert np.all(self._gain_map >= 0)
        self._de = de
        self._di = di
        self._set_transfer_funcs(baseline=rebalance_fic)
        self.__solve_fixed_point()
        self.compute_jacobian()
        if bold:
            self.moments_method(bold=True, gsr=gsr)

    def reset_syn_state(self):
        """ Reset state variables to their steady-state values. """
        self._I_I = np.copy(self._I_I_ss)
        self._I_E = np.copy(self._I_E_ss)
        self._r_I = np.copy(self._r_I_ss)
        self._r_E = np.copy(self._r_E_ss)
        self._S_I = np.copy(self._S_I_ss)
        self._S_E = np.copy(self._S_E_ss)

    @staticmethod
    def affine(x, m, b):
        """Affine transform x; that is, return ``y = m * x + b``.

        Parameters
        ----------
        x : np.ndarray
            values to affine transform
        b : float
            The intercept term
        m : float
            The scaling factor

        """
        return m * np.asarray(x) + b

    # Properties

    @property
    def sim_results(self):
        """
        Numerical simulation results.

        Returns
        -------
        lsd.model.sim.Sim
            Simulation instance

        """
        if self._sim is None:
            raise ValueError("No simulation data found")
        return self._sim

    @property
    def bw(self):
        """
        Hemodynamic model class instance.

        Returns
        -------
        lsd.lib.model.hemo.Balloon object
        """
        return self._bw

    @property
    def Q(self):
        """
        Returns
        -------
        np.ndarray
            Synaptic noise covariance matrix
        """
        return np.copy(self._Q)

    @property
    def cov(self):
        """
        Synaptic covariance matrix (approximated analytically).

        Returns
        -------
        np.ndarray
            Covariance matrix of linearized fluctuations about the fixed point.
        """
        return np.copy(self._cov)

    @property
    def var(self):
        """
        Synaptic variances (for excitatory gating variable S_E).

        Returns
        -------
        np.ndarray
        """
        return np.copy(np.diag(self._cov))

    @property
    def corr(self):
        """
        Synaptic FC matrix (approximated analytically).

        Returns
        -------
        np.ndarray
            FC matrix of linearized fluctuations about the fixed point.
        """
        return np.copy(self._corr)

    @property
    def cov_bold(self):
        """
        BOLD covariance matrix (approximated analytically).

        Returns
        -------
        np.ndarray
            Covariance matrix of linearized fluctuations about the fixed point.
        """
        return np.copy(self._bw.cov)

    @property
    def corr_bold(self):
        """
        BOLD FC matrix (approximated analytically).

        Returns
        -------
        np.ndarray
            FC matrix of linearized fluctuations about the fixed point.
        """
        return np.copy(self._bw.corr)

    @property
    def bold_fcz(self):
        fc = self._bw.corr
        fc[np.where(np.eye(180))] = 0
        return np.arctanh(fc)

    @property
    def jacobian(self):
        """
        Synaptic Jacobian matrix (i.e., S_E and S_I).

        Returns
        -------
        np.ndarray
            Jacobian of linearized fluctuations about fixed point. 
        """
        return np.copy(self._A)

    @property
    def evals(self):
        """
        Eigenvalues of synaptic Jacobian matrix.

        Returns
        -------
        np.ndarray

        """
        assert self._A is not None
        return eig(self._A)[0]

    @property
    def evecs(self):
        """
        Left eigenvectors of the synaptic Jacobian matrix.
        Returns
        -------
        np.ndarray
        """
        assert self._A is not None
        return eig(self._A)[1]

    @property
    def nareas(self):
        """
        Number of cortical areas, i.e. nodes.

        Returns
        -------
        np.ndarray

        """
        return self._nareas

    @property
    def sc(self):
        """
        Empirical structural connectivity matrix.

        Returns
        -------
        np.ndarray

        """
        return np.copy(self._sc)

    @property
    def sigma(self):
        """
        Variance of the independent noise input to each node.

        Returns
        -------
        float

        """
        return self._sigma

    @sigma.setter
    def sigma(self, s):
        """
        Set noise variance.

        Parameters
        ----------
        s : float
            noise variance.

        """
        self._sigma = s
        self._Q = np.identity(2 * self._nareas) * self._sigma * self._sigma
        self._bw = Balloon(
            nareas=self._nareas, syn_noise_var=self._sigma, linearize=False)

    @property
    def state(self):
        """
        State of all synaptic variables (i.e., the inhibitory and excitatory
        currents, rates, and gating fractions).

        Returns
        -------
        np.ndarray
            All state variables arranged into 6 rows by N_nodes columns.
            Rows are I_I, I_E, r_I, r_E, S_I, S_E.

        """
        x = np.vstack((self._I_I, self._I_E, self._r_I,
                       self._r_E, self._S_I, self._S_E))
        return x

    @property
    def steady_state(self):
        """
        Steady-state value of all synaptic state variables.

        Returns
        -------
        np.ndarray
            Steady-state variables, shape 6 rows by N_nodes columns.
            Rows are, respectively, I_I, I_E, r_I, r_E, S_I, S_E.
        """
        return np.vstack((self._I_I_ss, self._I_E_ss, self._r_I_ss,
                          self._r_E_ss, self._S_I_ss, self._S_E_ss))

    @property
    def S_E_ss(self):
        """
        Steady-state excitatory synaptic gating variables.

        Returns
        -------
        np.ndarray

        """
        return self._S_E_ss

    @property
    def w_EE(self):
        """
        Local recurrent excitation strength.

        Returns
        -------
        np.ndarray

        """
        return self._w_EE

    @property
    def w_IE(self):
        """
        Local excitatory to inhibitory connection strengths.

        Returns
        -------
        np.ndarray

        """
        return self._w_IE

    @property
    def G(self):
        """
        Global coupling strength.

        Returns
        -------
        float

        """
        return self._G

    @G.setter
    def G(self, g):
        """
        Set global coupling strength to a new value. Feedback inhibition is
        rebalanced, such that excitatory firing rates are ~3Hz; the new
        fixed point is found; and the synaptic Jacobian matrix is recomputed
        and evaluated at the system's fixed point.

        Parameters
        ----------
        g : float
            Global coupling strength.

        """
        self._G = g
        self._K_EE = (self._w_EE * self._J_NMDA * self._id) + (
                self._G * self._J_NMDA * self._sc)
        self._compute_FIC()
        self.__solve_fixed_point()
        self.compute_jacobian()

    @property
    def w_EI(self):
        """
        Local inhibitory to excitatory connection strength (i.e., feedback
        inhibition strength).

        Returns
        -------
        np.ndarray

        """
        return self._w_EI

    @property
    def h_map(self):
        """
        Heterogeneity map.

        Returns
        -------
        np.ndarray

        """
        return self._hmap

    @property
    def htr2a_map(self):
        """
        HTR2A map.

        Returns
        -------
        np.ndarray

        """
        return np.copy(self._gain_map)

    @property
    def I_ext(self):
        """
        External input currents.

        Returns
        -------
        np.ndarray

        """
        return self._I_ext

    @property
    def is_unstable(self):
        """
        System's dynamically stability (determine via eigendecomposition of the
        system's Jacobian matrix).

        Returns
        -------
        bool

        """
        return self._is_unstable

    @property
    def J_NMDA(self):
        """
        Effective NMDA conductance.

        Returns
        -------
        float

        """
        return self._J_NMDA

    @property
    def A(self):
        """ Alias for Model.jacobian """
        return np.copy(self._A)

    @property
    def sim(self):
        """
        Numerical simulation output.

        Returns
        -------
        lsd.lib.model.sim.Sim instance
        """
        assert self._sim is not None
        return self._sim
