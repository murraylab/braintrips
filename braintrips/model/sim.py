import pickle
from os.path import exists


class Sim:
    """ 
    Class derived from Model base class to time evolve
    system via Euler integration. Also includes hemodynamic
    response functionality.
    """
    def __init__(self):

        self.t = None
        self.t_points = None
        self.dt = None
        self.n_save = None
        self.seed = None
        self.I_I = None
        self.I_E = None
        self.r_I = None
        self.r_E = None
        self.S_I = None
        self.S_E = None
        self.x = None
        self.f = None
        self.v = None
        self.q = None
        self.y = None

        return

    def sim_dict(self):
        """
        Generates the dictionary to save simulated model variables.

          'I_I': inhibitory current [nA]
          'I_E': excitatory current [nA]
          'r_I': inhibitory rate [Hz]
          'r_E': excitatory rate [Hz]
          'S_I': inhibitory synaptic gating fraction
          'S_E': excitatory synaptic gating fraction
          'x':   vasodilatory signal
          'objective':   normalized inflow rate
          'v':   normalized blood volume
          'q':   normalized deoxyhemoglobin content
          'y':   BOLD signal (% change) 
        
        """

        sim = dict()
        sim['t'] = self.t
        sim['dt'] = self.dt
        sim['t_points'] = self.t_points
        sim['n_save'] = self.n_save
        sim['seed'] = self.seed
        sim['I_I'] = self.I_I
        sim['I_E'] = self.I_E
        sim['r_I'] = self.r_I
        sim['r_E'] = self.r_E
        sim['S_I'] = self.S_I
        sim['S_E'] = self.S_E
        sim['x'] = self.x
        sim['objective'] = self.f
        sim['v'] = self.v
        sim['q'] = self.q
        sim['y'] = self.y

        return sim

    def load_sim(self, sim_file):
        """
        Reconstructs simulation outputs from pickle file, if necessary
        """
        if exists(sim_file):
            sim_dict = pickle.load(open(sim_file, "rb"))
            for key, attr in sim_dict.items():
                self.__setattr__(key, attr)
            print("Simulation loaded.")

        return

    def time_series(self, var_type='S_E'):
        """
        Returns the simulated time series for all nodes
        for a given variable type, with each row corresponding
        to a unique node, and each column representing a point
        in time.

        Parameters
        ----------
        var_type: str
            One of the following:
            'I_I': inhibitory current [nA]
            'I_E': excitatory current [nA]
            'r_I': inhibitory rate [Hz]
            'r_E': excitatory rate [Hz]
            'S_I': inhibitory synaptic gating fraction
            'S_E': excitatory synaptic gating fraction
            'x':   vasodilatory signal
            'objective':   normalized inflow rate
            'v':   normalized blood volume
            'q':   normalized deoxyhemoglobin content
            'y':   BOLD signal (% change)

        """

        syn_keys = ['I_I', 'I_E', 'r_I', 'r_E', 'S_I', 'S_E']
        BOLD_keys = ['x', 'objective', 'v', 'q', 'y']

        if not self.has_sim:
            raise Exception("No simulation to use!")

        if var_type in syn_keys:
            return self.__getattribute__(var_type)
        elif var_type in BOLD_keys:
            if self.has_BOLD:
                return self.__getattribute__(var_type)
            else:
                raise Exception("No BOLD data in this simulation.")
        else:
            raise Exception('Unrecognized variable type.')

    @property
    def has_sim(self):
        return True if self.S_E is not None else False

    @property
    def has_BOLD(self):
        return True if self.y is not None else False
