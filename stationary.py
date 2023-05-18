import calfem
import numpy as np
import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.utils as cfu
import logging
import matplotlib.pyplot as plt
from plantml import *
import time


class TemperetureFEA:
    def __init__(self, geom: cfg.Geometry, isolated_id: int, convection_id: int, heated_id: int,
                 copper_id: int, nylon_id: int, dofs_per_node: int, el_type: int, el_size_factor: float) -> None:
        self.ABS_ZERO = 273.15
        self.geom = geom
        self.isolated = isolated_id
        self.convection = convection_id
        self.heated = heated_id
        self.copper = copper_id
        self.nylon = nylon_id
        self.dofs_per_node = dofs_per_node
        self.el_type = el_type
        self.el_size_factor = el_size_factor

    def create_mesh(self) -> None:
        self.mesh = cfm.GmshMesh(self.geom)
        # Type of mesh (linear triangular)
        self.mesh.elType = self.el_type
        # Temperature is one dof per node
        self.mesh.dofsPerNode = self.dofs_per_node
        # Factor that changes element sizes
        self.mesh.elSizeFactor = self.el_size_factor
        self.mesh.return_boundary_elements = True

        # bdofs are the boundary dofs! Dict with key = marker ID, value = element index
        self.coords, self.edof, self.dofs, self.bdofs, self.elementmarkers, self.belems = self.mesh.create()

    def show_mesh(self):
        cfv.figure()
        cfv.drawMesh(coords=self.coords, edof=self.edof,
                     dofs_per_node=self.dofs_per_node, el_type=self.el_type, filled=True)
        cfv.showAndWait()

    def show_geometry(self):
        cfv.draw_geometry(self.geom)
        cfv.showAndWait()

    def create_matrices(self, heat_flow: float, env_temp: float, convection_coeff: float,
                        Dcopper: np.array, Dnylon: np.array, thickness: float) -> None:
        # Implementing a CALFEM solver
        self.nDofs = np.size(self.dofs)
        self.ex, self.ey = cfc.coordxtr(self.edof, self.coords, self.dofs)
        # Global stiffness matrix
        self.K = np.zeros((self.nDofs, self.nDofs))
        # Global convection matrix
        self.Kc = np.zeros((self.nDofs, self.nDofs))
        # Boundary conditions vector (Convection + Neumann)
        self.fb = np.zeros((self.nDofs, 1))

        # The heated part has Neumann conditions
        for elem in self.belems[self.heated]:
            # The boundary dofs of this element
            elem_dofs = elem["node-number-list"]
            # Compute the line segment length between the nodes
            boundary_len = np.linalg.norm(
                self.coords[elem_dofs[1] - 1] - self.coords[elem_dofs[0] - 1])
            # Element bounday conditions
            fbe = -heat_flow * boundary_len * thickness / 2 * np.ones((2, 1))
            # Assemble into global boundary vector
            self.fb[elem_dofs[0] - 1] += fbe[0]
            self.fb[elem_dofs[1] - 1] += fbe[1]

        # The convection part of fb and convection matrix
        for elem in self.belems[self.convection]:
            # Same as for Neumann
            elem_dofs = elem["node-number-list"]
            boundary_len = np.linalg.norm(
                self.coords[elem_dofs[1] - 1] - self.coords[elem_dofs[0] - 1])
            print(boundary_len)
            fbe = convection_coeff * (env_temp + self.ABS_ZERO) * thickness * \
                boundary_len / 2 * np.ones((2, 1))
            # Assemble into global fb
            self.fb[elem_dofs[0] - 1] += fbe[0]
            self.fb[elem_dofs[1] - 1] += fbe[1]

            # Element convection matrix
            Kce = convection_coeff * boundary_len * thickness / \
                6 * np.array([[2, 1], [1, 2]])
            # Assemble into global convection matrix
            self.Kc[elem_dofs[0] - 1, elem_dofs[0] - 1] += Kce[0, 0]
            self.Kc[elem_dofs[0] - 1, elem_dofs[1] - 1] += Kce[0, 1]
            self.Kc[elem_dofs[1] - 1, elem_dofs[0] - 1] += Kce[1, 0]
            self.Kc[elem_dofs[1] - 1, elem_dofs[1] - 1] += Kce[1, 1]

        # Assemble the stiffness matrix
        for eltopo, elx, ely, marker in zip(self.edof, self.ex, self.ey, self.elementmarkers):
            # Check material
            if marker == self.copper:
                D = Dcopper
            elif marker == self.nylon:
                D = Dnylon
            else:
                logging.warning("Potential error, no material found")

            # Element stiffness matrix
            Ke = cfc.flw2te(elx, ely, ep=[thickness], D=D, eq=None)
            cfc.assem(eltopo, self.K, Ke)

        # Dirichlet boundary conditions (not used)
        self.bc = np.array([], 'i')
        self.bcVal = np.array([], 'f')
        # self.bc, self.bcVal = cfu.applybc(
        #     self.bdofs, self.bc, self.bcVal, self.convection, 18. + self.ABS_ZERO, 0)

    def solve_stationary_problem(self, show_solution=True) -> np.array:
        a, _ = cfc.solveq(self.K + self.Kc, self.fb, self.bc, self.bcVal)
        # Convert back to Celsius
        a -= self.ABS_ZERO * np.ones((a.size, 1))

        if show_solution:
            cfv.figure(fig_size=(10, 10))
            cfv.draw_nodal_values_shaded(
                a, self.coords, self.edof, title="Temperature (Celsius)")
            cfv.colorbar()
            cfv.draw_mesh(self.coords, self.edof,
                          self.dofs_per_node, self.el_type)
            cfv.show_and_wait()

        return a

    def draw_arbitrary_solution(self, a: np.array) -> None:
        corrected_temp = a - self.ABS_ZERO * np.ones((self.nDofs, 1))
        cfv.figure(fig_size=(10, 10))
        cfv.draw_nodal_values_shaded(
            corrected_temp, self.coords, self.edof, title="Temperature (Celsius)")
        cfv.colorbar()
        cfv.draw_mesh(self.coords, self.edof, self.dofs_per_node, self.el_type)
        cfv.show_and_wait()

    def create_transient_matrix(self, c_copper: float, rho_copper: float, c_nylon: float, rho_nylon: float) -> None:
        """Assemble the tranient C-matrix using the plantml method"""
        self.C = np.zeros((self.nDofs, self.nDofs))

        for eltopo, elx, ely, marker in zip(self.edof, self.ex, self.ey, self.elementmarkers):
            # Check material
            if marker == self.copper:
                c = c_copper
                rho = rho_copper
            elif marker == self.nylon:
                c = c_nylon
                rho = rho_nylon
            else:
                logging.warning("Potential error, no material found")

            # Element C-matrix
            Ce = plantml(elx, ely, c * rho)
            # Assemble global C-matrix
            cfc.assem(eltopo, self.C, Ce)

    def implicit_integrator(self, step_size: float, end_time: float, a0: np.array) -> list[np.array]:
        """Do implicit time integration on a given FE temperature problem"""
        # Intermediate matrix to avoid computing inverse twice
        inter_mat = np.linalg.inv(self.C + step_size * (self.K + self.Kc))
        A = step_size * np.dot(inter_mat, self.fb)
        B = inter_mat * self.C
        # Here we store the solutions for the different time steps
        a_vecs = [a0]
        current_time = 0.

        while current_time <= end_time:
            current_time += step_size
            a_vecs.append(A + np.dot(B, a_vecs[-1]))

        return a_vecs
