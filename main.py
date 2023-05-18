import calfem
import numpy as np
import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.utils as cfu
import logging
import matplotlib.pyplot as plt
from stationary import *

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
thickness = 5e-3                    # bunda
Dcopper = np.identity(2) * 385.     # Constitutive matrix for copper
Dnylon = np.identity(2) * .26       # Constitutive matrix for nylon
rho_copper = 8930.                  # Copper density
rho_nylon = 1100.                   # Nylon density
c_copper = 386.                     # Copper specific heat
c_nylon = 1500.                     # Nylon specific heat

# Solver properties
el_type = 2
dofs_per_node = 1
el_size_factor = .02


def createGeometry():
    """Creating the geometry of our gripper"""
    g = cfg.Geometry()
    g.point([0, 0.3*L])  # point 0
    g.point([0, 0.4*L])  # point 1
    g.point([0, 0.5*L])  # point 2
    g.point([0.1*L, 0.5*L])  # point 3
    g.point([0.1*L, 0.4*L])  # point 3
    g.point([0.4*L, 0.4*L])  # point 4
    g.point([0.45*L, 0.35*L])  # point 5
    g.point([0.45*L, 0.05*L])  # point 6
    g.point([0.9*L, 0.3*L])  # point 7
    g.point([L, 0.3*L])  # point 8
    g.point([L, 0.25*L])  # point 9
    g.point([0.9*L, 0.25*L])  # point 10
    g.point([0.45*L, 0])
    g.point([0.35*L, 0])
    g.point([0.35*L, 0.3*L])
    g.point([0.15*L, 0.3*L])
    g.point([0.15*L, 0.15*L])  # point 15
    g.point([0.1*L, 0.15*L])
    g.point([0.1*L, 0.3*L])
    N = len(g.points)
    # Copper part, assign appropriate boundary condition markers
    for i in range(N):
        if i == 0 or i == 2 or i == 9:
            g.spline([i, (i+1) % N], marker=isolated)
        elif i == 1:
            g.spline([i, (i+1) % N], marker=heated)
        elif i < 13:
            g.spline([i, (i+1) % N], marker=convection)
        else:
            g.spline([i, (i+1) % N])
    # Nylon part
    g.point([0, 0])
    g.spline([19, 0], marker=isolated)
    g.spline([19, 13], marker=isolated)
    # Define the different regions
    g.surface(list(range(19)), marker=copper)
    g.surface(list(range(13, 21)), marker=nylon)
    return g


if __name__ == "__main__":
    geom = createGeometry()
    FEM = TemperetureFEA(geom, isolated, convection, heated,
                         copper, nylon, dofs_per_node, el_type, el_size_factor)
    FEM.create_mesh()
    FEM.create_matrices(h, env_temp, alpha_conv, Dcopper, Dnylon, thickness)
    # FEM.show_mesh()
    a = FEM.solve_stationary_problem(show_solution=True)

    #FEM.create_transient_matrix(c_copper, rho_copper, c_nylon, rho_nylon)
    #a0 = (env_temp + FEM.ABS_ZERO) * np.ones((FEM.nDofs, 1))
    #a_vecs = FEM.implicit_integrator(0.1, 10., a0)

    # FEM.draw_arbitrary_solution(a_vecs[-1])

    # for a in a_vecs:
    #     FEM.draw_arbitrary_solution(a)
