import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
from stationary import StatTempFEA
import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
from main import createGeometry
from main import isolated, convection, heated, copper, nylon, dofs_per_node, el_size_factor, el_type, h, env_temp, alpha_conv, Dnylon, Dcopper, thickness, c_copper, c_nylon, rho_copper, rho_nylon
from matplotlib.animation import FileMovieWriter, FFMpegFileWriter

#fig, ax = plt.subplots()
#xdata, ydata = [], []
#ln, = ax.plot([], [], 'ro')


def f():
    geom = createGeometry()
    # cfv.draw_geometry(geom)
    # cfv.show_and_wait()
    FEM = StatTempFEA(geom, isolated, convection, heated,
                      copper, nylon, dofs_per_node, el_type, el_size_factor)
    FEM.create_mesh()
    FEM.create_matrices(h, env_temp, alpha_conv, Dcopper, Dnylon, thickness)
    # FEM.show_mesh()
    # FEM.show_geometry()
    # a = FEM.solve_stationary_problem(show_solution=True)

    # FEM.draw_selected_dofs(FEM.bdofs[heated])

    FEM.create_transient_matrix(
        thickness, c_copper, rho_copper, c_nylon, rho_nylon)
    a0 = np.ones((FEM.nDofs, 1)) * env_temp
    a_vecs = FEM.implicit_integrator(100, 1., a0)

    return a_vecs, FEM


if __name__ == "__main__":

    a_vecs, FEM = f()
    for j in range(len(a_vecs)):
        FEM.update_animation(a_vecs[:, j])
        plt.plot()
        plt.pause(0.01)
    plt.show()
