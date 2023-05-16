import calfem
import numpy as np
import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.utils as cfu
import logging

L = 0.005 # standard length
isolated = 1
heated = 2
copper = 3
nylon = 4
thickness = [5e-3] # bunda
Dcopper = np.identity(2)*385 # Constitutive matrix
Dnylon = np.identity(2)*0.26 ## Constitutive matrix
l = 5.0     # length
h = 1.0     # height
t = 0.2     # thickness
v = 0.35    # Poisson
E = 2.1e9   # Young modulus
ptype = 1   # plane stress



def createGeometry():
    """STEP 1 - Creating geometry"""
    g = cfg.Geometry()
    g.point([0, 0.3*L]) # point 0
    g.point([0, 0.4*L]) # point 1
    g.point([0, 0.5*L]) # point 2
    g.point([0.1*L, 0.5*L]) # point 3
    g.point([0.1*L, 0.4*L]) # point 3
    g.point([0.4*L, 0.4*L]) # point 4
    g.point([0.45*L, 0.35*L]) # point 5
    g.point([0.45*L,0.05*L]) # point 6
    g.point([0.9*L, 0.3*L]) # point 7
    g.point([L, 0.3*L]) # point 8
    g.point([L, 0.25*L]) # point 9
    g.point([0.9*L, 0.25*L]) # point 10
    g.point([0.45*L, 0])
    g.point([0.35*L, 0]) 
    g.point([0.35*L, 0.3*L])
    g.point([0.15*L, 0.3*L])
    g.point([0.15*L, 0.15*L])  #point 15
    g.point([0.1*L, 0.15*L])
    g.point([0.1*L, 0.3*L])
    g.point([0, 0])
    N = len(g.points)
    for i in range(0, N-2):
        if i == 2:
            g.spline([i, (i+1)%N], marker = heated) # line 0 # marker = 1e5
        else:
            g.spline([i, (i+1)%N], marker = isolated) # line 0

    g.spline([18, 0], marker = isolated)
    g.spline([0, 19], marker = isolated)
    g.spline([13, 19], marker = isolated)
    g.surface(list(range(19)), marker = copper, )
    g.surface(list(range(13, 21)), marker = nylon) 
    return g


def createMesh(g):
    """STEP 2 - Creating mesh"""
    mesh = cfm.GmshMesh(g)
    mesh.elType = 2           # Type of mesh
    mesh.dofsPerNode = 1       # Factor that changes element sizes
    mesh.elSizeFactor = 0.02    # Factor that changes element sizes

    coords, edof, dofs, bdofs, elementmarkers = mesh.create()


    """STEP 3- Show mesh"""
    cfv.figure()
    cfv.drawMesh(coords=coords,edof=edof,dofs_per_node=mesh.dofsPerNode,el_type=mesh.elType,filled=True)
    cfv.showAndWait()
    return coords, edof, dofs, bdofs, elementmarkers


g = createGeometry()

#cfv.draw_geometry(s1)
#¨cfv.draw_geometry(g)
cfv.showAndWait()

coords, edof, dofs, bdofs, elementmarkers = createMesh(g)


# Implementing a CALFEM solver
nDofs = np.size(dofs)
ex, ey = cfc.coordxtr(edof, coords, dofs) 
K = np.zeros([nDofs,nDofs]) # global stifness matrix



for eltopo, elx, ely, mark in zip(edof, ex, ey, elementmarkers):
    if mark == copper:
        D = Dcopper
    elif mark == nylon:
        D = Dnylon
    else:
        logging.warning("Potential error, no material found")

    
    Ke = cfc.flw2te(elx, ely, ep= thickness, D=Dcopper, eq=None)
    cfc.assem(eltopo, K, Ke)

# boundary conditions
bc = np.array([],'i')
bcVal = np.array([],'f')


fl = 0
fb = np.zeros(nDofs)

bc, bcVal = cfu.applybc(bdofs, bc, bcVal, left_support, 0.0, 0)
bc, bcVal = cfu.applybc(bdofs, bc, bcVal, right_support, 0.0, 2)

