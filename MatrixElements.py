import numpy as np
from math import gamma, factorial
from scipy.special import erf
from WaveFunction import WaveFunction


class MatrixElements:
    def __init__(self, wf_params):
        self.l, self.N = wf_params[:2]
        self.wf = WaveFunction(wf_params)
        eta = self.wf.eta
        self.eta_add = np.add.outer(eta, eta)
        self.conj_eta_add = np.add.outer(eta.conj(), eta)
        self.eta_mult = np.multiply.outer(eta, eta)
        self.conj_eta_mult = np.multiply.outer(eta.conj(), eta)
        self.N_cos_ij = np.multiply.outer(self.wf.N_cos, self.wf.N_cos)

    def overlap_matrix(self):
        AL = np.power(1 / self.eta_add, self.l + 1.5)
        BL = np.power(1 / self.conj_eta_add, self.l + 1.5)
        return self.calc_mat_elems(AL, BL) * gamma(self.l + 1.5) / 2

    def kinetic_pot(self, const1, mass):
        AH0 = np.divide(self.eta_mult, np.power(self.eta_add, self.l + 2.5))
        BH0 = np.divide(self.conj_eta_mult, np.power(self.conj_eta_add, self.l + 2.5))
        return self.calc_mat_elems(AH0, BH0) * const1 / mass * (self.l + 1.5) * gamma(self.l + 1.5)

    def nuclear_pot(self, potential_params):
        V0, Vso, S, J, R0, Rso, a0, aso = potential_params
        LS = J * (J + 1) - self.l * (self.l + 1) - S * (S + 1)
        r_step = 0.1
        r = np.arange(1e-6, 15, r_step)
        V1 = V0 * self.woods_saxon(a0, R0, r)
        V2 = 2 * Vso * LS / aso * np.exp((r - Rso) / aso) * np.power(self.woods_saxon(aso, Rso, r), 2) / r
        return np.einsum("ir, r, jr -> ij", self.wf.phi(r), (V1 + V2) * r**2, self.wf.phi(r)) * r_step

    def coulomb_pot(self, z1, z2, const2, Rc):
        Vc_0_Rc = self.V_coul_from_0_to_Rc(Rc)
        Vc_Rc_Inf = self.V_coul_from_Rc_to_Inf(Rc)
        return z1 * z2 * const2 * (Vc_0_Rc + Vc_Rc_Inf)

    def V_coul_from_0_to_Rc(self, Rc):
        AVc = 3 * self.integral_0_b(Rc, 2 * self.l + 2, self.eta_add) - \
              1 / Rc**2 * self.integral_0_b(Rc, 2 * self.l + 4, self.eta_add)
        BVc = 3 * self.integral_0_b(Rc, 2 * self.l + 2, self.conj_eta_add) - \
              1 / Rc**2 * self.integral_0_b(Rc, 2 * self.l + 4, self.conj_eta_add)
        return self.calc_mat_elems(AVc, BVc) / 2 / Rc

    def V_coul_from_Rc_to_Inf(self, Rc):
        # $\int_b^\Inf = \int_0^\Inf - \int_0^b$
        AVc = np.power(1 / self.eta_add, self.l + 1) * factorial(self.l) / 2 - \
              self.integral_0_b(Rc, 2 * self.l + 1, self.eta_add)
        BVc = np.power(1 / self.conj_eta_add, self.l + 1) * factorial(self.l) / 2 - \
              self.integral_0_b(Rc, 2 * self.l + 1, self.conj_eta_add)
        return self.calc_mat_elems(AVc, BVc)

    def calc_mat_elems(self, A, B):
        return (A + A.conj() + B + B.conj()).real * self.N_cos_ij / 4

    @staticmethod
    def woods_saxon(a, R, r):
        return 1 / (1 + np.exp((r - R) / a))

    @staticmethod
    def integral_0_b(b, n, a):
        if n == 0:
            return 0.5 * np.sqrt(np.pi / a) * erf(np.sqrt(a) * b)
        if n == 1:
            return (1 - np.exp(-a * b**2)) / 2 / a
        return (n - 1) / 2 / a * MatrixElements.integral_0_b(b, n - 2, a) - b**(n - 1) / 2 / a * np.exp(-a * b**2)

if __name__ == "__main__":
    wf_params = [1, 10, 1, 0.1, np.pi / 4]
    mat_elems = MatrixElements(wf_params)
    print(mat_elems.overlap_matrix())
