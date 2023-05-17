import calfem
import numpy as np
import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.utils as cfu
import logging
import matplotlib.pyplot as plt

"""PARAMETERS"""
# Geometric properties
L = 0.005       # Standard length
h = -1e5         # Heat supply
# Marker IDs
isolated = 1
heated = 2
convection = 3
copper = 4
nylon = 5

# Material properties
thickness = [5e-3]                  # bunda
Dcopper = np.identity(2) * 385.     # Constitutive matrix for copper
Dnylon = np.identity(2) * 0.26      # Constitutive matrix for nylon
alpha_conv = 40.                    # Heat transfer coefficient
abs_zero = -273.15                  # Absolute zero in Celsius
env_temp = 18. - abs_zero           # Temperature of the environment (Kelvin)


def createGeometry():
    """STEP 1 - Creating geometry"""
    # Creating the geometry of our gripper
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
        if i == 0:
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


def createMesh(g):
    """STEP 2 - Creating mesh"""
    mesh = cfm.GmshMesh(g)
    mesh.elType = 2           # Type of mesh (linear triangular)
    mesh.dofsPerNode = 1      # Temperature is one dof per node
    mesh.elSizeFactor = 0.02  # Factor that changes element sizes
    mesh.return_boundary_elements = True

    # bdofs are the boundary dofs! Dict with key = marker ID, value = element index
    coords, edof, dofs, bdofs, elementmarkers, belems = mesh.create()

    """STEP 3 - Show mesh"""
    # cfv.figure()
    # cfv.drawMesh(coords=coords, edof=edof,
    #              dofs_per_node=mesh.dofsPerNode, el_type=mesh.elType, filled=True)
    # cfv.showAndWait()
    return coords, edof, dofs, bdofs, elementmarkers, belems


def creatingStiffnessMatrix():
    g = createGeometry()
    # cfv.draw_geometry(s1)
    # cfv.draw_geometry(g)
    # cfv.showAndWait()

    coords, edof, dofs, bdofs, elementmarkers, belems = createMesh(g)
    # Implementing a CALFEM solver
    nDofs = np.size(dofs)
    ex, ey = cfc.coordxtr(edof, coords, dofs)
    # Global stiffness matrix
    K = np.zeros((nDofs, nDofs))
    # Global convection matrix
    Kc = np.zeros((nDofs, nDofs))
    # Boundary conditions vector (Convection + Neumann)
    fb = np.zeros((nDofs, 1))

    # Assembling convection boundary conditions

    # The heated part has Neumann conditions
    for elem in belems[heated]:
        # The boundary dofs of this element
        dofs = elem["node-number-list"]
        # Compute the line segment length between the nodes
        boundary_len = np.linalg.norm(
            coords[dofs[1] - 1] - coords[dofs[0] - 1])
        # Element bounday conditions
        fbe = -h * boundary_len / 2 * np.ones((2, 1))
        # Assemble into global boundary vector
        fb[dofs[0] - 1] = fbe[0]
        fb[dofs[1] - 1] = fbe[1]

    # The convection part of fb and convection matrix
    for elem in belems[convection]:
        # Same as for Neumann
        dofs = elem["node-number-list"]
        boundary_len = np.linalg.norm(
            coords[dofs[1] - 1] - coords[dofs[0] - 1])
        fbe = alpha_conv * env_temp * boundary_len / 2 * np.ones((2, 1))
        # Assemble into global fb
        fb[dofs[0] - 1] = fbe[0]
        fb[dofs[1] - 1] = fbe[1]

        # Element convection matrix
        Kce = alpha_conv * boundary_len / 6 * np.array([[2, 1], [1, 2]])
        # Assemble into global convection matrix
        Kc[dofs[0] - 1, dofs[0] - 1] = Kce[0, 0]
        Kc[dofs[0] - 1, dofs[1] - 1] = Kce[0, 1]
        Kc[dofs[1] - 1, dofs[0] - 1] = Kce[1, 0]
        Kc[dofs[1] - 1, dofs[1] - 1] = Kce[1, 1]

    # Assemble the stiffness matrix and the convection matrix
    for eltopo, elx, ely, mark in zip(edof, ex, ey, elementmarkers):
        # Check material
        if mark == copper:
            D = Dcopper
        elif mark == nylon:
            D = Dnylon
        else:
            logging.warning("Potential error, no material found")

        # Element stiffness matrix
        Ke = cfc.flw2te(elx, ely, ep=thickness, D=D, eq=None)
        # print(eltopo)
        cfc.assem(eltopo, K, Ke)

    # Boundary Dirichlet conditions
    bc = np.array([], 'i')
    bcVal = np.array([], 'f')

    # plt.spy(K)
    # plt.show()

    # Funkar endast för Dirichlet-villkor! Neumann räknas ut manuellt och sätts in i fb
    # Den säger bara att dessa a-värden är satta till ett bestämt värde
    # bc, bcVal = cfu.applybc(bdofs, bc, bcVal, isolated, 1000.0, 0)
    # bc, bcVal = cfu.applybc(bdofs, bc, bcVal, heated, h)

    # Add convection matrix when returning
    return K + Kc, fb, bc, bcVal, coords, edof


K, fb, bc, bcVal, coords, edof = creatingStiffnessMatrix()

a, _ = cfc.solveq(K, fb, bc, bcVal)

# Convert back to Celsius
a += abs_zero * np.ones((a.size, 1))

cfv.figure(fig_size=(10, 10))
cfv.draw_nodal_values_shaded(a, coords, edof, title="Temperature")
cfv.colorbar()
cfv.show_and_wait()
