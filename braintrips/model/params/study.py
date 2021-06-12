from braintrips.utils import loading

metric = loading  # similarity metric
w_ee = (1.4, 0)  # (min, scale) of self-recurrence strength
w_ie = (1., 0)  # (min, scale) of I->E strength
global_coupling = 0.85
