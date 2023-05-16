import calfem
import numpy as np
import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis as cfv


"""STEP 2"""
# Define truss structure
NELEM , NDOF = 3, 8
L = 1
ex = np.array ([[-L, 0], [0, 0], [L, 0]])
ey = np.array ([ [0, -L], [0, -L], [0, -L]])
edof = np.array ([[1, 2, 7, 8],[3, 4, 7, 8],[5, 6, 7, 8]])

# Bar parameters
E = 210e9
A = np.array ([1e-4, 1e-4, 1e-4])

"""STEP 3"""
# Define stiffness matrix
K = np.zeros ((NDOF , NDOF))

# Compute stiffness matrix
kei = np.zeros ((4, 4))
for i in np.arange (0, NELEM):
    kei = cfc.bar2e(ex[i, :], ey[i, :], [E, A[i]])
    cfc.assem(edof[i, :], K, kei)

"""STEP 4"""
# Boundary conditions
F = np.zeros ((NDOF , 1))
F[7] = -10e3
bc_dof = np.array ([1, 2, 3, 4, 5, 6])
bc_val = np.array ([0, 0, 0, 0, 0, 0])

"""STEP 5"""
# Solve BVP
a, r = cfc.solveq(K, F, bc_dof , bc_val)
# Print values
print("Displacements:")
print(a)
print("Reaction forces:")
print(r)