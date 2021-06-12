"""Parameters for the dynamic mean field model,
taken from table 3 of Deco et al. 2014 (J Neurosci) .

"""

# Excitatory gating variables
a_E = 310.    # [nC^-1]
b_E = 125.    # [Hz]
d_E = 0.16    # [s]
tau_E = 0.1   # [s] (NMDA)
W_E = 1.0     # excitatory external input weight

# Inhibitory gating variables
a_I = 615.    # [nC^-1]
b_I = 177.    # [Hz]
d_I = 0.087   # [s]
tau_I = 0.01  # [s] (GABA)
W_I = 0.7     # inhibitory external input weight

# Other variables from text
I_task = 0.02        # [nA] for (non-RSN case, not currently implemented)
gamma = 0.641        # kinetic conversion factor (typo in text)
J_NMDA = 0.15        # [nA] (excitatory synaptic coupling)
I0 = 0.382           # [nA] (overall effective external input)
sigma = 0.01         # [nA] (noise amplitude)

# Additional long-range projecting current term (used in single-node case)
I_ext = 0.0  # [nA]

# Synaptic weight connections
w_EE = 1.4     # E->E self-coupling
w_II = 1.0     # I->I self-coupling
w_IE = 1.0     # E->I coupling
w_EI = 1.0     # I->I coupling

# Steady-state solutions in isolated case
r_I_ss = 3.9218448633  # Hz
r_E_ss = 3.0773270642  # Hz
I_I_ss = 0.2528951325  # nA
I_E_ss = 0.3773805650  # nA
S_I_ss = 0.0392184486  # dimensionless
S_E_ss = 0.1647572075  # dimensionless
