import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import spsolve, eigsh
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FEASolver:
    def __init__(self, wing_geometry, material):
        self.wing = wing_geometry
        self.material = material
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

    def assemble_global_stiffness(self, thicknesses, thetas):
        ### Assemble the global stiffness matrix with variable fiber angles
        try:
            K_data = []
            K_rows = []
            K_cols = []
            for e, element in enumerate(self.wing.elements):
                Ke = self.element_stiffness(element, thicknesses[e], thetas[e])
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

    def assemble_global_mass(self, thicknesses):
        ### Assemble the global mass matrix
        try:
            M_data = []
            M_rows = []
            M_cols = []
            for e, element in enumerate(self.wing.elements):
                Me = self.element_mass(element, thicknesses[e])
                element_dof = [node * 2 + i for node in element for i in range(2)]
                for i, dof_i in enumerate(element_dof):
                    for j, dof_j in enumerate(element_dof):
                        M_data.append(Me[i, j])
                        M_rows.append(dof_i)
                        M_cols.append(dof_j)
            M = coo_matrix((M_data, (M_rows, M_cols)), shape=(self.total_dof, self.total_dof))
            logging.debug("Assembled mass matrix")
            return M
        except Exception as e:
            logging.error(f"Mass matrix assembly failed: {e}")
            raise

    def element_stiffness(self, element, thickness, theta):
        ### Compute the stiffness matrix for a quadrilateral element
        try:
            if thickness < 1e-6:
                raise ValueError(f"Invalid thickness {thickness} for element {element}")
            D = self.material.get_stiffness_matrix(theta)
            gauss_points = [(-1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(3), -1/np.sqrt(3)),
                            (-1/np.sqrt(3), 1/np.sqrt(3)), (1/np.sqrt(3), 1/np.sqrt(3))]
            weights = [1, 1, 1, 1]
            Ke = np.zeros((8, 8))
            for gp, w in zip(gauss_points, weights):
                B, detJ = self.shape_functions_and_derivatives(element, gp[0], gp[1])
                Ke += thickness * detJ * w * (B.T @ D @ B)
            return Ke
        except Exception as e:
            logging.error(f"Element stiffness computation failed for element {element}: {e}")
            raise

    def element_mass(self, element, thickness):
        ### Compute the mass matrix for a quadrilateral element
        try:
            if thickness < 1e-6:
                raise ValueError(f"Invalid thickness {thickness} for element {element}")
            rho = self.material.density
            gauss_points = [(-1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(3), -1/np.sqrt(3)),
                            (-1/np.sqrt(3), 1/np.sqrt(3)), (1/np.sqrt(3), 1/np.sqrt(3))]
            weights = [1, 1, 1, 1]
            Me = np.zeros((8, 8))
            for gp, w in zip(gauss_points, weights):
                N, detJ = self.shape_functions_mass(element, gp[0], gp[1])
                Me += rho * thickness * detJ * w * (N.T @ N)
            return Me
        except Exception as e:
            logging.error(f"Element mass computation failed for element {element}: {e}")
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

    def shape_functions_mass(self, element, xi, eta):
        ### Compute shape functions for mass matrix
        try:
            N = [
                0.25 * (1 - xi) * (1 - eta),
                0.25 * (1 + xi) * (1 - eta),
                0.25 * (1 + xi) * (1 + eta),
                0.25 * (1 - xi) * (1 + eta)
            ]
            N_mat = np.zeros((2, 8))
            for i in range(4):
                N_mat[0, 2*i] = N[i]
                N_mat[1, 2*i + 1] = N[i]
            coords = self.wing.nodes[element][:, :2]
            J = np.zeros((2, 2))
            dN_dxi = [
                [-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)],
                [-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)]
            ]
            for i in range(4):
                J[0, 0] += dN_dxi[0][i] * coords[i, 0]
                J[1, 0] += dN_dxi[1][i] * coords[i, 0]
                J[0, 1] += dN_dxi[0][i] * coords[i, 1]
                J[1, 1] += dN_dxi[1][i] * coords[i, 1]
            detJ = np.linalg.det(J)
            if detJ <= 0:
                logging.error(f"Invalid Jacobian for element {element}: detJ={detJ}")
                raise ValueError(f"Non-positive Jacobian determinant: {detJ}")
            return N_mat, detJ
        except Exception as e:
            logging.error(f"Shape function computation failed for element {element}: {e}")
            raise

    def apply_boundary_conditions(self, K, F, M=None):
        ### Apply fixed boundary conditions
        try:
            fixed_nodes = [i for i in range(self.wing.num_elements_y + 1)]
            fixed_dofs = [node * 2 + i for node in fixed_nodes for i in range(2)]
            K = K.tolil()
            if M is not None:
                M = M.tolil()
            for dof in fixed_dofs:
                K[dof, :] = 0
                K[:, dof] = 0
                K[dof, dof] = 1
                F[dof] = 0
                if M is not None:
                    M[dof, :] = 0
                    M[:, dof] = 0
                    M[dof, dof] = 1
            K = K.tocsr()
            M = M.tocsr() if M is not None else None
            logging.debug("Applied boundary conditions")
            return K, F, M
        except Exception as e:
            logging.error(f"Boundary condition application failed: {e}")
            raise

    def apply_loads(self):
        ### Apply distributed aerodynamic load
        try:
            F = np.zeros(self.total_dof)
            total_lift = 3e3
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
        ### Check buckling constraints
        try:
            E = self.material.E1
            L = self.wing.thickness
            for e in self.wing.spar_elements:
                t = thicknesses[e]
                if t < 1e-6:
                    logging.warning(f"Invalid thickness {t} for spar element {e}")
                    return False
                I = t**3 * self.wing.element_size_x / 12
                P_cr = np.pi**2 * E * I / (L**2)
                P = 3e3 * self.wing.element_size_x / len(self.wing.spar_elements)
                if P > P_cr:
                    logging.debug(f"Spar buckling failure: P={P}, P_cr={P_cr}")
                    return False
            b = self.wing.element_size_x
            for e in range(len(self.wing.elements)):
                if e not in self.wing.spar_elements:
                    t = thicknesses[e]
                    if t < 1e-6:
                        logging.warning(f"Invalid thickness {t} for skin element {e}")
                    k = 4
                    D = self.material.E1 * t**3 / (12 * (1 - self.material.nu12**2))
                    sigma_cr = k * np.pi**2 * D / (b**2 * t)
                    if stresses is None or stresses[e] > sigma_cr:
                        logging.debug(f"Skin buckling failure: stress={stresses[e] if stresses is not None else 'N/A'}, sigma_cr={sigma_cr}")
                        return False
            logging.debug("Buckling check passed")
            return True
        except Exception as e:
            logging.error(f"Buckling check failed: {e}")
            return False

    def check_frequency(self, K, M):
        ### Calculate the lowest natural frequency
        try:
            K_dense = K.toarray()
            M_dense = M.toarray()
            cond_K = np.linalg.cond(K_dense)
            cond_M = np.linalg.cond(M_dense)
            logging.debug(f"Stiffness matrix condition number: {cond_K:.2e}, Mass matrix condition number: {cond_M:.2e}")
            eigenvalues, _ = eigsh(K, k=1, M=M, sigma=0, which='LM', tol=1e-3)
            freq = np.sqrt(np.abs(eigenvalues[0])) / (2 * np.pi)
            logging.debug(f"Frequency calculated: f={freq:.2f} Hz, eigenvalue={eigenvalues[0]:.2e}")
            return freq
        except Exception as e:
            logging.error(f"Frequency calculation failed: {e}")
            return 0.0

    def solve(self, thicknesses, thetas):
        ### Perform finite element analysis (FEA)
        try:
            if np.any(thicknesses < 1e-6):
                logging.error(f"Invalid thicknesses detected: {thicknesses}")
                raise ValueError("Invalid thicknesses detected")
            K = self.assemble_global_stiffness(thicknesses, thetas)
            M = self.assemble_global_mass(thicknesses)
            F = self.apply_loads()
            K, F, M = self.apply_boundary_conditions(K, F, M)
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
                D = self.material.get_stiffness_matrix(thetas[e])
                strain = B @ u_e
                stress = D @ strain
                von_mises = np.sqrt(stress[0]**2 + stress[1]**2 - stress[0]*stress[1] + 3*stress[2]**2)
                stresses.append(von_mises)
            max_stress = np.max(stresses)
            max_displacement = np.max(np.abs(u))
            stress_feasible = max_stress <= self.material.sigma_y
            buckling_feasible = self.check_buckling(thicknesses, stresses)
            displacement_feasible = max_displacement <= 0.15
            freq = self.check_frequency(K, M)
            is_feasible = stress_feasible and buckling_feasible and displacement_feasible
            logging.debug(f"FEA results: max_stress={max_stress/1e6} MPa, max_displacement={max_displacement} m, freq={freq:.2f} Hz, feasible={is_feasible}")
            return u, stresses, max_stress, max_displacement, freq, is_feasible
        except Exception as e:
            logging.error(f"FEA solve failed: {e}")
            raise
