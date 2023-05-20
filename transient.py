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


class TransientTempFEA:
    def __init__(self, K: np.array, Kc: np.array, f: np.array, nDofs: int, edof: np.array,
                 ex: np.array, ey: np.array, elementmarkers: np.array, copper_id: int, nylon_id: int) -> None:
        self.K, self.Kc, self.f = K, Kc, f
        self.nDofs = nDofs
        self.edof = edof
        self.ex, self.ey = ex, ey
        self.elementmarkers = elementmarkers
        self.copper, self.nylon = copper_id, nylon_id

    def create_transient_matrix(self, thickness: float, c_copper: float, rho_copper: float, c_nylon: float, rho_nylon: float) -> None:
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
            Ce = plantml(elx, ely, thickness * c * rho)
            # Assemble global C-matrix
            cfc.assem(eltopo, self.C, Ce)

    def implicit_integrator(self, end_time: float, step_size: float, init_temps: np.array) -> list[np.array]:
        """Do implicit time integration on a given FE temperature problem"""

        tt = np.arange(0, end_time, step_size)
        temps = np.empty((self.nDofs, tt.size))
        temps[:, 0] = np.reshape(init_temps, (self.nDofs))

        for t in range(tt.size - 1):
            current_temps = temps[:, t]
            f_array = np.reshape(self.f, (self.nDofs))
            next_temps = np.linalg.solve(
                self.C + step_size * (self.K + self.Kc), np.dot(self.C, current_temps) + step_size * f_array)
            temps[:, t + 1] = next_temps

        return temps

    def compute_90_percent_of_max(self, max_temp: float, temps: np.array, step_size: float) -> float:
        current_max_temp = -273.15
        time_point = 0

        while current_max_temp < .9 * max_temp:
            current_max_temp = max(temps[:, time_point])
            time_point += 1

        return time_point * step_size
