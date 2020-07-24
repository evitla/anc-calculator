from numpy import random, round
from scipy.optimize import minimize
from NuclearSystem import NuclearSystem

def energy_function_to_minimize(vars, system_params1, system_params2, wf_params, potential_params, E_exp_p, E_exp_n):
    potential_params[0] = vars[0]   # V0
    A = system_params1[0]
    R0 = vars[1] * A**(1/3)
    system_params1[-1] = R0
    potential_params[4:6] = R0, R0
    system_params2[1] = 0   # neutron charge
    system_p = NuclearSystem(wf_params, system_params1, potential_params)
    system_n = NuclearSystem(wf_params, system_params2, potential_params)
    system_p.calculate_energy(mode="bound_energy")
    system_n.calculate_energy(mode="bound_energy")
    return abs(system_p.bound_energy - E_exp_p) + abs(system_n.bound_energy - E_exp_n)

def find_V0_and_r0(system_params, wf_params, potential_params, E_exp_p, E_exp_n ):
    V0_min, V0_max = -80, -30
    r0_min, r0_max = 1, 1.5
    system_params2 = system_params.copy()
    while True:
        V0 = random.uniform(low=V0_min, high=V0_max, size=(1,))
        r0 = random.uniform(low=r0_min, high=r0_max, size=(1,))
        vars = [V0, r0]
        res = minimize(energy_function_to_minimize, vars,
                       args=(system_params, system_params2, wf_params, potential_params, E_exp_p, E_exp_n),
                       method="nelder-mead", tol=1e-4, options={"disp": False})
        if energy_function_to_minimize(res.x, system_params, system_params2, wf_params, potential_params, E_exp_p, E_exp_n) < 1e-3:
            break
    V0, r0 = res.x
    return V0, round(r0, 5)
