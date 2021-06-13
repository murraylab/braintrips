from braintrips.data import sc, fc_pla, htr2a, group_dgbc
from braintrips.model.params.study import w_ee, w_ie
from braintrips.model.dmf import Model
from braintrips.utils import loading
from braintrips.model.params.study import coupling as study_coupling
from braintrips.model.params.synaptic import r_E_ss, r_I_ss
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np

triu = np.triu_indices(180, k=1)


def make_new_model(coupling_strength, gsr=True):
    model_ = Model(
        sc=sc, g=coupling_strength, w_ee=w_ee,
        w_ie=w_ie, htr2a_map=htr2a, de=0, di=0
    )
    model_.moments_method(bold=True, gsr=gsr)
    return model_


# Candidate global coupling strength values
coupling_vals = np.arange(0.01, 1, 0.01)

# Sweep over global coupling strengths and compute Spearman rank correlation
# between off-diagonal elements of simulated and empirical placebo FC matrices
scores = np.empty(coupling_vals.size)
for i, gc in enumerate(coupling_vals):
    model = make_new_model(coupling_strength=gc)
    if model.is_unstable:  # Check dynamical stability
        print("Unstable G = ", gc)
        continue
    mfcz = model.bold_fcz  # r-to-Z transformed BOLD FC matrix
    sim = spearmanr(mfcz[triu], fc_pla[triu])[0]
    print(model.G, sim)
    scores[i] = sim
print(scores.max(), coupling_vals[scores.argmax()])

# Note that, as reported in the paper, we use a slightly less-than-optimal value
#  of global coupling to move the system further from the edge of dynamical
#  instability


delta = 0.003
dx = np.arange(0, 0.03+1E-9, 0.003)
nx = dx.size

# Instantiate empty arrays
dgbc_maps = np.array([np.nan] * nx * nx * 180).reshape((nx, nx, 180))
gbc_sim = np.array([np.nan] * nx * nx).reshape((nx, nx))
re = np.array([np.nan] * nx * nx).reshape((nx, nx))
ri = np.array([np.nan] * nx * nx).reshape((nx, nx))

# Instantiate new model
model = make_new_model(coupling_strength=study_coupling)

# Baseline FC and GBC
baseline_mfcz = model.bold_fcz
baseline_gbc = model.gbc(baseline_mfcz)

# Main loop
for ii, di in enumerate(dx):
    for jj, de in enumerate(dx):

        # Update gain modulation parameters
        model.update_gain_params(
            de=de, di=di, rebalance_fic=False, bold=True, gsr=True)

        # Check dynamical stability
        if model.is_unstable:
            continue

        # Modulated FC and GBC
        modulated_fcz = model.bold_fcz
        modulated_gbc = model.gbc(modulated_fcz)

        # Delta GBC and FC
        dmgbc = modulated_gbc - baseline_gbc
        dmfcz = modulated_fcz - baseline_mfcz

        # Steady-state firing rates
        exc_rate = model.steady_state[3, :].mean()
        inh_rate = model.steady_state[2, :].mean()

        # Store maps and similarity scores
        dgbc_maps[ii, jj, :] = dmgbc
        if not np.allclose(modulated_gbc, baseline_gbc):
            gbc_sim[ii, jj] = loading(dmgbc, group_dgbc)
        re[ii, jj] = exc_rate
        ri[ii, jj] = inh_rate

        print(de, di, gbc_sim[ii, jj])

de_vals, di_vals = np.meshgrid(dx, dx)
de_opt = de_vals[gbc_sim == np.nanmax(gbc_sim)]
di_opt = di_vals[gbc_sim == np.nanmax(gbc_sim)]
print("Optimal params:", de_opt, di_opt)
print("Best model fit:", np.nanmax(gbc_sim))

# Reproduce Figure 2A
fig, ax = plt.subplots()

asmasked = lambda m: np.ma.masked_array(m, np.isnan(m))
mres = asmasked(gbc_sim)
im = ax.imshow(mres, cmap='viridis', interpolation='nearest', aspect=1,
               origin='lower', extent=[0, 0.03+delta, 0, 0.03+delta],
               vmin=-1, vmax=4)

cbar = plt.colorbar(
    im, ax=ax, ticks=[], pad=0.05, fraction=0.1, shrink=1)
cbar.ax.tick_params(width=0, length=0)
cbar.outline.set_visible(False)

pos = np.linspace(0, 0.03, 11)
labels = ['0%', '', '', '', '', '1.5%', '', '', '', '', '3%']
ax.set_xticks(pos + delta / 2)
ax.set_yticks(pos + delta / 2)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

cbar.ax.text(0, 4.5, '4', ha='left', va='bottom', fontsize=10)
cbar.ax.text(0, -1, '-1', ha='left', va='top', fontsize=10)

ax.xaxis.set_tick_params(pad=2)
ax.yaxis.set_tick_params(pad=2)

ax.set_xlabel("Excitatory gain increase", fontsize=12)
ax.set_ylabel("Inhibitory gain increase", fontsize=12)

iopt = mres.flatten().argmax()
ev, iv = np.meshgrid(dx, dx)
de_opt = ev.flatten()[iopt]
di_opt = iv.flatten()[iopt]
ax.scatter(de_opt + 0.0015, di_opt + 0.0015, s=150, c='k', marker='*')
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

plt.show()


# ----------

# E/I perturbation

baseline_ei_ratio = r_E_ss / r_I_ss
eipert = (re/ri) / baseline_ei_ratio
sim_copy = asmasked(np.copy(gbc_sim))
x = eipert.flatten()
y = sim_copy.flatten()
eimax = x[y.argmax()]

fig2, ax = plt.subplots()

ax.scatter(x, y, s=50, clip_on=False, zorder=2, c=y, cmap='viridis')
ax.scatter(eimax, 4.55, s=100, c='k', marker='*', clip_on=False)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_ylim(-1, 4)
ax.set_yticks([-1, 4])
# detach(ax, which='both')
ax.set_xlabel("Relative E/I ratio", fontsize=12)
ax.set_ylabel("Model-empirical\nloading", fontsize=12)

plt.show()
