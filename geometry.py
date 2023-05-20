import calfem
import numpy as np
import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.utils as cfu
import matplotlib.pyplot as plt


def createTempGeometry(L: float, copper: int, nylon: int, heated: int, convection: int, isolated: int) -> cfg.Geometry:
    """Creating the geometry of our gripper for computing temperatures"""
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
        if i == 0 or i == 2 or i == 9 or i == 12:
            g.spline([i, (i + 1) % N], marker=isolated)
        elif i == 1:
            g.spline([i, (i + 1) % N], marker=heated)
        elif i < 12:
            g.spline([i, (i + 1) % N], marker=convection)
        else:
            g.spline([i, (i + 1) % N])
    # Nylon part
    g.point([0, 0])
    g.spline([19, 0], marker=isolated)
    g.spline([19, 13], marker=isolated)
    # Define the different regions
    g.surface(list(range(19)), marker=copper)
    g.surface(list(range(13, 21)), marker=nylon)
    return g


def createStressGeometry(L: float, copper: int, nylon: int, heated: int, convection: int, isolated: int) -> cfg.Geometry:
    """Creating the geometry of our gripper for computing displacements and stresses"""
    # TODO!!
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
    # for i in range(N):
    #     if i == 0 or i == 2 or i == 9 or i == 12:
    #         g.spline([i, (i + 1) % N], marker=isolated)
    #     elif i == 1:
    #         g.spline([i, (i + 1) % N], marker=heated)
    #     elif i < 12:
    #         g.spline([i, (i + 1) % N], marker=convection)
    #     else:
    #         g.spline([i, (i + 1) % N])
    # # Nylon part
    # g.point([0, 0])
    # g.spline([19, 0], marker=isolated)
    # g.spline([19, 13], marker=isolated)
    # # Define the different regions
    # g.surface(list(range(19)), marker=copper)
    # g.surface(list(range(13, 21)), marker=nylon)
    return g
