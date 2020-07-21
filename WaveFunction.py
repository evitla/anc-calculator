from math import gamma
import numpy as np

class WaveFunction:
    def __init__(self, wf_params):
        self.l, N, t, a0, b = wf_params
        alpha = a0 * np.tan((2 * np.arange(N) + 1) * np.pi / 4 / N)**t
        self.eta = alpha * (1 + 1j * b)

        l15 = self.l + 1.5
        b_sq1 = (b**2 + 1)**(l15/2)
        self.N_cos = 2 * (2 * alpha)**(l15/2) * b_sq1 / np.sqrt(gamma(l15) * (b_sq1 + np.cos(l15 * np.arctan(b))))

    def psi(self, r):
        return np.einsum("i,ir->ir", self.N_cos / 2, r**self.l * np.exp(-np.multiply.outer(self.eta, r**2)))

    def phi(self, r):
        return (self.psi(r).conj() + self.psi(r)).real

wf_params = [1, 10, 1, 0.1, np.pi / 4]
wf = WaveFunction(wf_params)
r = np.arange(0, 10, 0.1)
print(wf.phi(r)[5])
