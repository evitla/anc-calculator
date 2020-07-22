from math import gamma
import numpy as np
import matplotlib.pyplot as plt


class WaveFunction:
    def __init__(self, wf_params):
        self.l, N, t, a0, b = wf_params
        alpha = a0 * np.tan((2 * np.arange(N) + 1) * np.pi / 4 / N)**t
        self.eta = alpha * (1 + 1j * b)
        l15 = self.l + 1.5
        b_sq1 = (b**2 + 1)**(l15/2)
        self.N_cos = 2 * (2 * alpha)**(l15/2) * np.sqrt(b_sq1 / gamma(l15) / (b_sq1 + np.cos(l15 * np.arctan(b))))

    def psi(self, r):
        return np.einsum("i,ir->ir", self.N_cos / 2, r**self.l * np.exp(-np.multiply.outer(self.eta, r**2)))

    def phi(self, r):
        return (self.psi(r).conj() + self.psi(r)).real

    def draw_phi(self, r, *j, f="wavefunctions"):
        fig, ax = plt.subplots(figsize=(10, 6))
        font = {
            "family": "Arial",
            "color": "black",
            "weight": "normal",
            "size": 14
        }
        if f == "density":
            p, q = j
            plt.plot(r, self.phi(r)[p] * self.phi(r)[q], lw=2)
            ylab = f"$\phi_{p}^*\phi_{q}$"
        else:
            [plt.plot(r, self.phi(r)[i], lw=2, label=f"$\phi_{i}$") for i in j]
            plt.legend(fontsize=font["size"] - 2)
            ylab = "$\phi_i$"
        ax.tick_params(axis="both", labelsize=font["size"] - 2)
        plt.xlabel("$r$, fm", fontdict=font)
        plt.ylabel(ylab, fontdict=font)
        plt.show()


if __name__ == "__main__":
    wf_params = [1, 10, 1, 0.1, np.pi / 4]
    wf = WaveFunction(wf_params)
    r = np.arange(0, 20, 0.1)
    wf.draw_phi(r, *range(5))
    wf.draw_phi(r, 1, 5, f="density")
