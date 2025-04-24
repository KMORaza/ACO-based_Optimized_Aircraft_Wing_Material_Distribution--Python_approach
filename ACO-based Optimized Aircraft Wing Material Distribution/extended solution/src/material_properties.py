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
        self.density = density
        self.sigma_y = sigma_y
        logging.info(f"Initialized material: E1={E1/1e9} GPa, density={density} kg/m^3, nu12={nu12}, G12={G12/1e9} GPa")

    def get_stiffness_matrix(self, theta=0):
        ### Compute the stiffness matrix for a given fiber angle (degrees)
        try:
            theta_rad = np.radians(theta)
            c = np.cos(theta_rad)
            s = np.sin(theta_rad)
            S = np.zeros((3, 3))
            S[0, 0] = 1 / self.E1
            S[1, 1] = 1 / self.E2
            S[0, 1] = -self.nu12 / self.E1
            S[1, 0] = -self.nu21 / self.E2
            S[2, 2] = 1 / self.G12
            T = np.array([
                [c**2, s**2, 2*c*s],
                [s**2, c**2, -2*c*s],
                [-c*s, c*s, c**2 - s**2]
            ])
            S_global = T.T @ S @ T
            Q = np.linalg.inv(S_global)
            logging.debug(f"Computed stiffness matrix for theta={theta} degrees")
            return Q
        except Exception as e:
            logging.error(f"Stiffness matrix computation failed: {e}")
            raise
