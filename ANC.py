import numpy as np
from mpmath import whitw
from NuclearSystem import NuclearSystem
from WaveFunction import WaveFunction


class ANC:
    def __init__(self, system_params, wf_params, potential_params, r):
        system = NuclearSystem(wf_params, system_params, potential_params)
        system.calculate_energy()
        self.bound_energy = system.energies[0]

        wf = WaveFunction(wf_params)
        radial_wavefunctions = r * np.matmul(system.C.transpose(), wf.phi(r))
        self.bound_wavefunction = np.abs(radial_wavefunctions[0,:])

        k = np.sqrt(-2 * self.bound_energy * system.mass / system.const1)
        eta = system.mass * system.z1 * system.z2 * system.const2 / k / system.const1

        self.whitw_ = np.array([whitw(-eta, wf.l + 0.5, 2 * k * r_i) for r_i in r], dtype=float)

    def function(self):
        return self.bound_wavefunction / self.whitw_


if __name__ == "__main__":
    A, Z = 99, 49
    L, S, J = 4, 0.5, 4.5
    V0, Vso, r0, a0 = -57.526593, -6, 1.22791, 0.65
    R0 = r0 * A**(1/3)
    N, t, alpha0, b = 80, 5, 0.1, np.pi / 12
    system_params = [A, 1, Z, R0]
    wf_params = [L, N, t, alpha0, b]
    potential_params = [V0, Vso, S, J, R0, R0, a0, a0]
    r = np.arange(0, 15, 0.1)
    anc_In99_p = ANC(system_params, wf_params, potential_params, r)
    print(anc_In99_p.bound_energy)
    print(anc_In99_p.function()[-10:])
