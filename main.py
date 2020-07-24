import numpy as np
import matplotlib.pyplot as plt
from energy_minimization import find_V0_and_r0
from ANC import ANC

A, Z = 99, 49
L, S, J = 4, 0.5, 4.5
E_exp_p, E_exp_n = -3.196, -17.410
a0, Vso = 0.65, -12
r0, V0 = 1, -80
R0 = r0 * A**(1/3)
N, t, alpha0, b = 80, 5, 0.1, np.pi / 12
system_params = [A, 1, Z, R0]
wf_params = [L, N, t, alpha0, b]
potential_params = [V0, Vso, S, J, R0, R0, a0, a0]
V0, r0 = find_V0_and_r0(system_params, wf_params, potential_params, E_exp_p, E_exp_n)
print(V0, r0)

R0 = r0 * A**(1/3)
potential_params[0], *potential_params[4:6] = V0, R0, R0
system_params_p = [A, 1, Z, R0]
system_params_n = [A, 0, Z, R0]

r = np.arange(0.1, 12.1, 0.1)
anc_p = ANC(system_params_p, wf_params, potential_params, r)
anc_n = ANC(system_params_n, wf_params, potential_params, r)
print(anc_p.bound_energy, anc_n.bound_energy)

anc_p_func = anc_p.function()
anc_n_func = anc_n.function()

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
font = {
    "family": "Arial",
    "color": "black",
    "weight": "normal",
    "size": 12
}
ax1.plot(r, anc_p_func, lw=3, label=r"$\langle^{99}_{49}X|^{100}_{50}Y\rangle$")
ax2.plot(r, anc_n_func, lw=3, label=r"$\langle^{99}_{49}X|^{100}_{49}X\rangle$")
for ax in (ax1, ax2):
    ax.legend(fontsize=font["size"])
    ax.tick_params(axis="both", labelsize=font["size"] - 2)
    ax.set_xlabel("$r$, fm", fontdict=font)
    ax.set_ylabel("$u_l(r)/W_{-\eta, l+1/2}(2kr)$", fontdict=font)
plt.show()
