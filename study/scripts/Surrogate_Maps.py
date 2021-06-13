from braintrips.data import sc, htr2a, group_dgbc, distmat
from braintrips.model.params.study import w_ee, w_ie
from braintrips.model.dmf import Model
from braintrips.utils import loading
from braintrips.model.params.study import coupling as study_coupling
from braintrips.utils import linearize_map
from braintrips.utils import nonparametric_pvalue

from brainsmash.mapgen.base import Base
import numpy as np

n_surr = 1000

triu = np.triu_indices(180, k=1)


def make_new_model(coupling_strength, gsr=True):
    model_ = Model(
        sc=sc, g=coupling_strength, w_ee=w_ee,
        w_ie=w_ie, htr2a_map=htr2a, de=0, di=0
    )
    model_.moments_method(bold=True, gsr=gsr)
    return model_


# Simulate spatial autocorrelation-preserving surrogate maps using BrainSMASH
gen = Base(x=htr2a, D=distmat, resample=True, seed=0)
surrogate_maps = gen(n=n_surr)

# Note that the absolute scale of these maps doesn't matter because they will
#  undergo the same sequence of transformations (linearization then shifting
#  and re-scaling to unit interval) as the empirical HTR2A map prior to being
#  used to modulate neural gain

# Establish baseline GBC (unaffected by the choice of gain modulatory map) and
# the test statistic
model = Model(
    sc=sc, g=study_coupling, w_ee=w_ee, w_ie=w_ie, htr2a_map=htr2a, de=0, di=0)
model.moments_method(bold=True, gsr=True)
baseline_mfcz = model.bold_fcz
baseline_gbc = model.gbc(baseline_mfcz)
model.update_gain_params(de=0.009, di=0, bold=True, gsr=True)
modulated_gbc = model.gbc(model.bold_fcz)
dgbc_model = modulated_gbc - baseline_gbc
test_stat = loading(dgbc_model, group_dgbc)

# Reparametrize model with spatial autocorrelation-preserving surrogate maps
# to construct null distribution of expected model-empirical loading
null_dist = np.empty(n_surr)
for i, x in enumerate(surrogate_maps):
    print(i)
    model = Model(sc=sc, g=study_coupling, w_ee=w_ee, w_ie=w_ie,
                  htr2a_map=linearize_map(x), de=0, di=0)
    model.moments_method(bold=True, gsr=True)
    # use best-fit params from empirical sweep to modulate gain
    model.update_gain_params(de=0.009, di=0, bold=True, gsr=True)
    surr_mfcz = model.bold_fcz
    mod_gbc_surr = model.gbc(surr_mfcz)
    dgbc_surr = mod_gbc_surr - baseline_gbc
    null_dist[i] = loading(dgbc_surr, group_dgbc)

p_surr_sa = nonparametric_pvalue(test_stat, null_dist, two_tailed=True)
print("Nonparametric p-value, SA-preserving surrogates (two-tailed):",
      p_surr_sa)
