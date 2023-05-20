import calfem
import numpy as np
import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.utils as cfu
import matplotlib.pyplot as plt
# Our files
from stationary import StatTempFEA
from transient import TransientTempFEA
from animation import animate
from geometry import createTempGeometry

"""PARAMETERS"""
# Geometric and boundary condition properties
L = 5e-3            # Standard length
h = -1e5            # Heat supply
alpha_conv = 40.    # Heat transfer coefficient
env_temp = 18.      # Temperature of the environment (Celsius)

# Marker IDs
isolated = 1    # Boundary on which q_n = 0 (includes symmetry axes)
heated = 2      # Boundary on which q_n = h
convection = 3  # Boundary on which q_n = alpha_conv * (T - env_temp)
copper = 5      # Region made of copper
nylon = 6       # Region made of nylon

# Material properties
thickness = 5e-3            # bunda
Dcopper = np.eye(2) * 385.  # Constitutive matrix for copper
Dnylon = np.eye(2) * .26    # Constitutive matrix for nylon
rho_copper = 8930.          # Copper density
rho_nylon = 1100.           # Nylon density
c_copper = 386.             # Copper specific heat
c_nylon = 1500.             # Nylon specific heat

# Solver properties
el_type = 2
dofs_per_node = 1
el_size_factor = .02

def a(show_solution=True):
    """TASK A: STATIONARY TEMPERATURE DISTRIBUTION"""
    geom = createTempGeometry(L, copper, nylon, heated, convection, isolated)
    StatFEM = StatTempFEA(geom, isolated, convection, heated,
                          copper, nylon, dofs_per_node, el_type, el_size_factor)
    StatFEM.create_mesh()
    # FEM.show_mesh()
    # FEM.show_geometry()
    StatFEM.create_matrices(h, env_temp, alpha_conv,
                            Dcopper, Dnylon, thickness)

    # Solve the statonary problem (K + Kc)a = fh + fc
    a = StatFEM.solve_stationary_problem(show_solution=show_solution)
    # What is the maximum nodal temperature?
    max_stat_temp = max(a)
    if show_solution:
        print(
            f"Maximum stationary temperature: {float(max_stat_temp)} deg. Celsius")
    return StatFEM, max_stat_temp

def b( do_animation = True):
    StatFEM, max_stat_temp = a(show_solution=False)
    """TASK B: TRANSIENT HEATFLOW"""
    # Numerical integration parameters
    end_time = 80.
    dt = .01

    TransFEM = TransientTempFEA(StatFEM.K, StatFEM.Kc, StatFEM.f, StatFEM.nDofs, StatFEM.edof,
                                StatFEM.ex, StatFEM.ey, StatFEM.elementmarkers, StatFEM.copper, StatFEM.nylon)
    # Calculate the C-matrix from the transient heatflow eq. C dot{a} + (K + Kc)a = fh + fc
    TransFEM.create_transient_matrix(
        thickness, c_copper, rho_copper, c_nylon, rho_nylon)
    # Initial temeprature is the environment temperature
    init_temp = np.ones((TransFEM.nDofs, 1)) * env_temp
    # The temperatures at each time point
    print(f"Doing numerical integration, please hold tight!")
    temps = TransFEM.implicit_integrator(end_time, dt, init_temp)
    # Animate the solution
    if do_animation:
        animate(StatFEM, temps)
    # Compute 90% of the maximum temperature
    time90 = TransFEM.compute_90_percent_of_max(max_stat_temp, temps, dt)
    print(f"Time to reach 90% of maximum: {time90} seconds")

def c():
    pass

if __name__ == "__main__":
    #a()
    b()
    #c()