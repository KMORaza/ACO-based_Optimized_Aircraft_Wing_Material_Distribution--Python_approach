import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Material:
    def __init__(self, E1=135e9, E2=10e9, G12=5e9, nu12=0.3, density=1600, sigma_y=1200e6):
        self.E1 = E1
        self.E2 = E2
        self.G12 = G12
        self.nu12 = nu12
        self.nu21 = nu12 * E2 / E1
        self.rho = density
        self.sigma_y = sigma_y
        logging.info(f"Initialized material: E1={E1/1e9} GPa, density={density} kg/m^3")

    def get_stiffness_matrix(self, theta=0):
        ### Compute the 2D plane stress stiffness matrix
        try:
            c = np.cos(np.radians(theta))
            s = np.sin(np.radians(theta))
            Q11 = self.E1 / (1 - self.nu12 * self.nu21)
            Q22 = self.E2 / (1 - self.nu12 * self.nu21)
            Q12 = self.nu12 * self.E2 / (1 - self.nu12 * self.nu21)
            Q66 = self.G12
            Q = np.array([[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]])
            T = np.array([[c**2, s**2, 2*c*s], [s**2, c**2, -2*c*s], [-c*s, c*s, c**2 - s**2]])
            Qbar = np.linalg.inv(T) @ Q @ T
            logging.debug("Computed stiffness matrix")
            return Qbar
        except Exception as e:
            logging.error(f"Stiffness matrix computation failed: {e}")
            raise
