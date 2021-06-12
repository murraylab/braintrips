"""
Hemodynamic parameters for Balloon-Windkessel model according to Obata et al
2004 (Neuroimage).

Updated k_{1,2,3} parameters derived from K. Stephan et al 2007 (Neuroimage)
for data collected in a 3T scanner.

"""

V0 = 0.02     # resting blood volume fraction
kappa = 0.65  # [s^-1] rate of signal decay
gamma = 0.41  # [s^-1] rate of flow-dependent elimination
tau = 0.98    # [s] hemodynamic transit time
alpha = 0.32  # Grubb's exponent
rho = 0.34    # resting oxygen extraction fraction

k1 = 3.72
k2 = 0.527
k3 = 0.53

# OLD METHOD
# k1 = 7 * rho
# k2 = 1.43 * rho
# k3 = 0.43


def param_dict():
    """ Return dictionary of fixed hemodynamic parameter values. """
    d = dict()
    d['V0'] = V0
    d['kappa'] = kappa
    d['gamma'] = gamma
    d['tau'] = tau
    d['alpha'] = alpha
    d['rho'] = rho
    d['k1'] = k1
    d['k2'] = k2
    d['k3'] = k3
    return d
