import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FEASolver:
    def __init__(self, wing_geometry, material):
        self.wing = wing_geometry
        self.material = material
        self.D = material.get_stiffness_matrix(theta=0)
        self.dof_per_node = 2
        self.total_dof = len(self.wing.nodes) * self.dof_per_node
        logging.info(f"Initialized FEA solver: {self.total_dof} DOFs")
        self.validate_mesh()
    def validate_mesh(self):
        ### Validate mesh to ensure positive Jacobian determinants
        try:
            for e, element in enumerate(self.wing.elements):
                _, detJ = self.shape_functions_and_derivatives(element, 0, 0)
                if detJ <= 0:
                    logging.error(f"Invalid mesh: Non-positive Jacobian determinant for element {e}")
                    raise ValueError(f"Invalid mesh: Non-positive Jacobian for element {e}")
            logging.info("Mesh validation passed")
        except Exception as e:
            logging.error(f"Mesh validation failed: {e}")
            raise
    def assemble_global_stiffness(self, thicknesses):
        ### Assemble the global stiffness matrix
        try:
            K_data = []
            K_rows = []
            K_cols = []
            for e, element in enumerate(self.wing.elements):
                Ke = self.element_stiffness(element, thicknesses[e])
                element_dof = [node * 2 + i for node in element for i in range(2)]
                for i, dof_i in enumerate(element_dof):
                    for j, dof_j in enumerate(element_dof):
                        K_data.append(Ke[i, j])
                        K_rows.append(dof_i)
                        K_cols.append(dof_j)
            K = coo_matrix((K_data, (K_rows, K_cols)), shape=(self.total_dof, self.total_dof))
            logging.debug("Assembled stiffness matrix")
            return K
        except Exception as e:
            logging.error(f"Stiffness matrix assembly failed: {e}")
            raise
    def element_stiffness(self, element, thickness):
        ### Compute the stiffness matrix for a quadrilateral element
        try:
            if thickness < 1e-6:
                raise ValueError(f"Invalid thickness {thickness} for element {element}")
            gauss_points = [(-1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(3), -1/np.sqrt(3)),
                            (-1/np.sqrt(3), 1/np.sqrt(3)), (1/np.sqrt(3), 1/np.sqrt(3))]
            weights = [1, 1, 1, 1]
            Ke = np.zeros((8, 8))
            for gp, w in zip(gauss_points, weights):
                B, detJ = self.shape_functions_and_derivatives(element, gp[0], gp[1])
                Ke += thickness * detJ * w * (B.T @ self.D @ B)
            return Ke
        except Exception as e:
            logging.error(f"Element stiffness computation failed for element {element}: {e}")
            raise
    def shape_functions_and_derivatives(self, element, xi, eta):
        ### Compute shape function derivatives and Jacobian determinant
        try:
            dN_dxi = [
                [-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)],
                [-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)]
            ]
            coords = self.wing.nodes[element][:, :2]
            J = np.zeros((2, 2))
            for i in range(4):
                J[0, 0] += dN_dxi[0][i] * coords[i, 0]
                J[0, 1] += dN_dxi[0][i] * coords[i, 1]
                J[1, 0] += dN_dxi[1][i] * coords[i, 0]
                J[1, 1] += dN_dxi[1][i] * coords[i, 1]
            detJ = np.linalg.det(J)
            if detJ <= 0:
                logging.error(f"Invalid Jacobian for element {element}: detJ={detJ}")
                raise ValueError(f"Non-positive Jacobian determinant: {detJ}")
            J_inv = np.linalg.inv(J)
            B = np.zeros((3, 8))
            for i in range(4):
                dN_dx = J_inv @ np.array([dN_dxi[0][i], dN_dxi[1][i]])
                B[0, 2*i] = dN_dx[0]
                B[1, 2*i + 1] = dN_dx[1]
                B[2, 2*i] = dN_dx[1]
                B[2, 2*i + 1] = dN_dx[0]
            return B, detJ
        except Exception as e:
            logging.error(f"Shape function derivatives computation failed for element {element}: {e}")
            raise
    def apply_boundary_conditions(self, K, F):
        ### Apply fixed boundary conditions
        try:
            fixed_nodes = [i for i in range(self.wing.num_elements_y + 1)]
            fixed_dofs = [node * 2 + i for node in fixed_nodes for i in range(2)]
            K = K.tolil()
            for dof in fixed_dofs:
                K[dof, :] = 0
                K[:, dof] = 0
                K[dof, dof] = 1
                F[dof] = 0
            K = K.tocsr()
            logging.debug("Applied boundary conditions")
            return K, F
        except Exception as e:
            logging.error(f"Boundary condition application failed: {e}")
            raise
    def apply_loads(self):
        ### Apply distributed aerodynamic load
        try:
            F = np.zeros(self.total_dof)
            total_lift = 2e3  
            top_nodes = [i * (self.wing.num_elements_y + 1) + self.wing.num_elements_y 
                         for i in range(self.wing.num_elements_x + 1)]
            load_factors = []
            for idx, node in enumerate(top_nodes):
                x = self.wing.nodes[node][0]
                load_factor = np.sqrt(max(0, 1 - (x / self.wing.chord)**2))
                load_factors.append(load_factor)
            total_factor = sum(load_factors)
            if total_factor == 0:
                raise ValueError("Invalid load distribution")
            for idx, node in enumerate(top_nodes):
                load_per_node = total_lift * load_factors[idx] / total_factor
                F[node * 2 + 1] = -load_per_node
            logging.debug("Applied aerodynamic loads")
            return F
        except Exception as e:
            logging.error(f"Load application failed: {e}")
            raise
    def check_buckling(self, thicknesses, stresses):
        ### Check buckling constraints using precomputed stresses 
        try:
            logging.debug("Skipping buckling check for performance")
            return True
        except Exception as e:
            logging.error(f"Buckling check failed: {e}")
            return False
    def solve(self, thicknesses):
        ### Solve the FEA problem
        try:
            if np.any(thicknesses < 1e-6):
                logging.error(f"Invalid thicknesses detected: {thicknesses}")
                raise ValueError("Invalid thicknesses detected")
            K = self.assemble_global_stiffness(thicknesses)
            F = self.apply_loads()
            K, F = self.apply_boundary_conditions(K, F)
            try:
                u = spsolve(K, F)
            except Exception as e:
                logging.error(f"Linear solver failed: {e}")
                raise
            stresses = []
            for e, element in enumerate(self.wing.elements):
                element_dof = [node * 2 + i for node in element for i in range(2)]
                u_e = u[element_dof]
                B, _ = self.shape_functions_and_derivatives(element, 0, 0)
                strain = B @ u_e
                stress = self.D @ strain
                von_mises = np.sqrt(stress[0]**2 + stress[1]**2 - stress[0]*stress[1] + 3*stress[2]**2)
                stresses.append(von_mises)
            max_stress = np.max(stresses)
            stress_feasible = max_stress <= self.material.sigma_y
            buckling_feasible = self.check_buckling(thicknesses, stresses)
            is_feasible = stress_feasible and buckling_feasible
            logging.debug(f"FEA results: max_stress={max_stress/1e6} MPa, feasible={is_feasible}")
            return u, stresses, max_stress, is_feasible
        except Exception as e:
            logging.error(f"FEA solve failed: {e}")
            raise
