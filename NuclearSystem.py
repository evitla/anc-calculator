import numpy as np
from MatrixElements import MatrixElements


class NuclearSystem:
    def __init__(self, wf_params, system_params, potential_params):
        self.wf_params = wf_params
        mass1 = 1 # nucleon mass
        mass2, self.z1, self.z2, self.Rc = system_params
        self.potential_params = potential_params
        self.mass = mass1 * mass2 / (mass1 + mass2)
        self.const1, self.const2 = 41.8016, 1.43996468

    def calculate_energy(self):
        mat_elems = MatrixElements(self.wf_params)
        L = mat_elems.overlap_matrix()
        H0 = mat_elems.kinetic_pot(self.const1, self.mass)
        Vs = mat_elems.nuclear_pot(self.potential_params)
        Vc = mat_elems.coulomb_pot(self.z1, self.z2, self.const2, self.Rc)

        Hc = H0 + Vc
        H = Hc + Vs

        down = np.linalg.cholesky(L)
        up = down.transpose()
        inv_down, inv_up = np.linalg.inv(down), np.linalg.inv(up)
        new_H = np.matmul(np.matmul(inv_down, H), inv_up)

        energies, new_H_eig_vectors = np.linalg.eig(new_H)
        H_eig_vectors = np.matmul(inv_up, new_H_eig_vectors)

        idx_sort = np.argsort(energies)
        self.energies = energies[idx_sort]
        self.C = H_eig_vectors[:, idx_sort]

if __name__ == "__main__":
    A, Z = 99, 49
    L, S, J = 4, 0.5, 4.5
    V0, Vso, r0, a0 = -57.526593, -6, 1.22791, 0.65
    R0 = r0 * A**(1/3)
    N, t, alpha0, b = 80, 5, 0.1, np.pi / 12
    system_params = [A, 1, Z, R0]
    wf_params = [L, N, t, alpha0, b]
    potential_params = [V0, Vso, S, J, R0, R0, a0, a0]
    In99_p = NuclearSystem(wf_params, system_params, potential_params)
    In99_p.calculate_energy()
    print(In99_p.energies)
