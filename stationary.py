import calfem
import numpy as np
import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.utils as cfu
import logging
import matplotlib.pyplot as plt
from plantml import plantml


class StatTempFEA:
    def __init__(self, geom: cfg.Geometry, isolated_id: int, convection_id: int, heated_id: int,
                 copper_id: int, nylon_id: int, dofs_per_node: int, el_type: int, el_size_factor: float) -> None:
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
        mesh = cfm.GmshMesh(self.geom)
        # Type of mesh (linear triangular)
        mesh.elType = self.el_type
        # Temperature is one dof per node
        mesh.dofsPerNode = self.dofs_per_node
        # Factor that changes element sizes
        mesh.elSizeFactor = self.el_size_factor
        mesh.return_boundary_elements = True

        # bdofs are the boundary dofs! Dict with key = marker ID, value = element index
        self.coords, self.edof, self.dofs, self.bdofs, self.elementmarkers, self.belems = mesh.create()

    def show_mesh(self):
        cfv.figure()
        cfv.drawMesh(coords=self.coords, edof=self.edof,
                     dofs_per_node=self.dofs_per_node, el_type=self.el_type, filled=True)
        cfv.showAndWait()

    def show_geometry(self):
        cfv.draw_geometry(self.geom)
        cfv.showAndWait()

    def draw_selected_dofs(self, selected_dofs: list):
        selected_dofs = [val - 1 for val in selected_dofs]
        selected_coords = self.coords[selected_dofs]
        cfv.draw_geometry(self.geom)
        cfv.drawMesh(coords=self.coords, edof=self.edof,
                     dofs_per_node=self.dofs_per_node, el_type=self.el_type, filled=False)
        plt.plot(selected_coords[:, 0], selected_coords[:, 1], 'o')
        plt.show()

    def create_matrices(self, heat_flow: float, env_temp: float, convection_coeff: float,
                        Dcopper: np.array, Dnylon: np.array, thickness: float) -> None:
        # Implementing a CALFEM solver
        self.nDofs = np.size(self.dofs)
        self.ex, self.ey = cfc.coordxtr(self.edof, self.coords, self.dofs)
        # Global stiffness matrix
        self.K = np.zeros((self.nDofs, self.nDofs))
        # Global convection matrix
        self.Kc = np.zeros((self.nDofs, self.nDofs))
        # Boundary conditions vector (convection)
        self.fc = np.zeros((self.nDofs, 1))
        # Boundary conditions vector (Neumann)
        self.fh = np.zeros((self.nDofs, 1))

        # The heated part has Neumann conditions
        for elem in self.belems[self.heated]:
            # The boundary dofs of this element
            elem_dofs = elem["node-number-list"]
            # Compute the line segment length between the nodes
            boundary_len = np.linalg.norm(
                self.coords[elem_dofs[1] - 1] - self.coords[elem_dofs[0] - 1])
            # Element bounday conditions
            fhe_val = -heat_flow * boundary_len * thickness / 2
            # Assemble into global boundary vector
            self.fh[elem_dofs[0] - 1] += fhe_val
            self.fh[elem_dofs[1] - 1] += fhe_val

        # Convection boundary conditions (vector and matrix)
        for elem in self.belems[self.convection]:
            # Same as for Neumann
            elem_dofs = elem["node-number-list"]

            boundary_len = np.linalg.norm(
                self.coords[elem_dofs[1] - 1] - self.coords[elem_dofs[0] - 1])
            fce_val = convection_coeff * env_temp * thickness * boundary_len / 2
            # Assemble into global fc
            self.fc[elem_dofs[0] - 1] += fce_val
            self.fc[elem_dofs[1] - 1] += fce_val

            # Element convection matrix
            common_value = convection_coeff * boundary_len * thickness
            # Assemble into global convection matrix
            self.Kc[elem_dofs[0] - 1, elem_dofs[0] - 1] += common_value / 3
            self.Kc[elem_dofs[0] - 1, elem_dofs[1] - 1] += common_value / 6
            self.Kc[elem_dofs[1] - 1, elem_dofs[0] - 1] += common_value / 6
            self.Kc[elem_dofs[1] - 1, elem_dofs[1] - 1] += common_value / 3

        self.f = self.fh + self.fc

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
            Ke = cfc.flw2te(elx, ely, [thickness], D)
            cfc.assem(eltopo, self.K, Ke)

        # Dirichlet boundary conditions (not used)
        self.bc = np.array([], 'i')
        self.bcVal = np.array([], 'f')
        # self.bc, self.bcVal = cfu.applybc(
        #     self.bdofs, self.bc, self.bcVal, self.convection, 18., 0)

    def solve_stationary_problem(self, show_solution=True) -> np.array:
        a = np.linalg.solve(self.K + self.Kc, self.fh + self.fc)

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
        cfv.figure(fig_size=(10, 10))
        cfv.draw_nodal_values_shaded(
            a, self.coords, self.edof, title="Temperature (Celsius)")
        cfv.colorbar()
        cfv.draw_mesh(self.coords, self.edof, self.dofs_per_node, self.el_type)
        cfv.show_and_wait()

    def update_animation(self, a: np.array) -> None:
        cfv.figure(fig_size=(10, 10))
        cfv.draw_nodal_values_shaded(
            a, self.coords, self.edof, title="Temperature (Celsius)")
        cfv.colorbar()
        cfv.draw_mesh(self.coords, self.edof, self.dofs_per_node, self.el_type)
