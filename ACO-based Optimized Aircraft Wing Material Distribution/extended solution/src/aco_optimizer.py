import numpy as np
import logging
import time
from wing_geometry import WingGeometry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ACOOptimizer:
    def __init__(self, wing_geometry, fea_solver, num_ants=20, max_iterations=30, rho=0.3, alpha=3.0, beta=2.0):
        self.wing = wing_geometry
        self.fea = fea_solver
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.num_elements = len(wing_geometry.elements)
        #### Thickness options
        self.thickness_options_skin = np.array([0.030, 0.035, 0.040, 0.045])  # 30, 35, 40, 45 mm
        self.thickness_options_spar = np.array([0.100, 0.120, 0.140, 0.160])  # 100, 120, 140, 160 mm
        #### Fiber angle options (degrees)
        self.theta_options = np.array([0, 30, 45, 60, 90])
        #### Young's modulus E1 options (GPa)
        self.E1_options = np.array([170e9, 200e9, 230e9])  # 170, 200, 230 GPa
        #### Density options (kg/m^3)
        self.density_options = np.array([1400, 1600, 1800])  # 1400, 1600, 1800 kg/m^3
        #### Poisson's ratio nu12 options
        self.nu12_options = np.array([0.25, 0.3, 0.35])  # 0.25, 0.3, 0.35
        #### Shear modulus G12 options (GPa)
        self.G12_options = np.array([8e9, 9e9, 10e9])  # 8, 9, 10 GPa
        #### Spar position options (x-coordinates in meters)
        self.spar_position_options = np.array([0.2, 1.0, 1.8])  # 0.2, 1.0, 1.8 m
        #### Ply layup sequence options (discrete angles)
        self.layup_options = np.array([0, 45, 90])  
        self.num_thickness_options = len(self.thickness_options_skin)
        self.num_theta_options = len(self.theta_options)
        self.num_E1_options = len(self.E1_options)
        self.num_density_options = len(self.density_options)
        self.num_nu12_options = len(self.nu12_options)
        self.num_G12_options = len(self.G12_options)
        self.num_spar_positions = len(self.spar_position_options)
        self.num_layup_options = len(self.layup_options)
        #### Pheromone trails
        self.pheromones_thickness = np.ones((self.num_elements, self.num_thickness_options)) * 0.1
        self.pheromones_theta = np.ones((self.num_elements, self.num_theta_options)) * 0.1
        self.pheromones_E1 = np.ones((self.num_elements, self.num_E1_options)) * 0.1
        self.pheromones_density = np.ones((self.num_elements, self.num_density_options)) * 0.1
        self.pheromones_nu12 = np.ones((self.num_elements, self.num_nu12_options)) * 0.1
        self.pheromones_G12 = np.ones((self.num_elements, self.num_G12_options)) * 0.1
        self.pheromones_spar = np.ones((2, self.num_spar_positions)) * 0.1
        self.pheromones_layup = np.ones((self.num_elements, self.num_layup_options)) * 0.1
        #### Stronger biases for thicker elements, non-zero angles, higher E1/G12, and layups
        self.pheromones_thickness[:, 2:] = 0.5  # 40, 45 mm (skin), 140, 160 mm (spars)
        self.pheromones_theta[:, 2:] = 0.6  # 45, 60, 90 degrees
        self.pheromones_E1[:, 1:] = 0.5  # 200, 230 GPa
        self.pheromones_G12[:, 1:] = 0.5  # 9, 10 GPa
        self.pheromones_layup[:, 1:] = 0.5  # 45, 90 degrees
        #### Heuristics (normalized)
        self.eta_thickness_skin = 1 / self.thickness_options_skin
        self.eta_thickness_skin /= np.sum(self.eta_thickness_skin)
        self.eta_thickness_spar = 1 / self.thickness_options_spar
        self.eta_thickness_spar /= np.sum(self.eta_thickness_spar)
        self.eta_theta = np.array([0.1, 0.2, 0.3, 0.2, 0.2])  # 45 degrees
        self.eta_theta /= np.sum(self.eta_theta)
        self.eta_E1 = 1 / (self.E1_options / 1e9)
        self.eta_E1 /= np.sum(self.eta_E1)
        self.eta_density = 1 / self.density_options
        self.eta_density /= np.sum(self.eta_density)
        self.eta_nu12 = 1 / self.nu12_options
        self.eta_nu12 /= np.sum(self.eta_nu12)
        self.eta_G12 = 1 / (self.G12_options / 1e9)
        self.eta_G12 /= np.sum(self.eta_G12)
        self.eta_spar = 1 / (self.spar_position_options + 1e-10)
        self.eta_spar /= np.sum(self.eta_spar)
        self.eta_layup = np.array([0.2, 0.4, 0.4])  # 45, 90 degrees
        self.eta_layup /= np.sum(self.eta_layup)

        self.best_weights = []
        self.feasible_counts = []
        self.stress_feasible_counts = []
        self.disp_feasible_counts = []
        self.buckling_feasible_counts = []
        self.buckling_penalties = []
        self.best_frequencies = []
        logging.info(f"Initialized ACO: {num_ants} ants, {max_iterations} iterations, {self.num_elements} elements, 2 spars")

    def normalize_probabilities(self, prob):
        """Normalize probabilities and handle invalid cases."""
        prob = np.clip(prob, 1e-10, np.inf)
        prob_sum = np.sum(prob)
        if prob_sum < 1e-10 or not np.isfinite(prob_sum):
            return np.ones(len(prob)) / len(prob)
        prob = prob / prob_sum
        if not np.isclose(np.sum(prob), 1, rtol=1e-5):
            logging.warning(f"Probability normalization adjusted: sum={np.sum(prob)}")
            prob = np.ones(len(prob)) / len(prob)
        return prob

    def construct_solution(self):
        ### Construct a solution (thickness, theta, E1, density, nu12, G12, spar positions, layup)
        try:
            thicknesses = np.zeros(self.num_elements)
            thetas = np.zeros(self.num_elements)
            E1s = np.zeros(self.num_elements)
            densities = np.zeros(self.num_elements)
            nu12s = np.zeros(self.num_elements)
            G12s = np.zeros(self.num_elements)
            spar_positions = np.zeros(2)
            layups = np.zeros(self.num_elements, dtype=int)
            for e in range(self.num_elements):
                eta_thickness = self.eta_thickness_spar if e in self.wing.spar_elements else self.eta_thickness_skin
                thickness_options = self.thickness_options_spar if e in self.wing.spar_elements else self.thickness_options_skin
                prob_thickness = (self.pheromones_thickness[e] ** self.alpha) * (eta_thickness ** self.beta)
                prob_thickness = self.normalize_probabilities(prob_thickness)
                thicknesses[e] = thickness_options[np.random.choice(self.num_thickness_options, p=prob_thickness)]
                prob_theta = (self.pheromones_theta[e] ** self.alpha) * (self.eta_theta ** self.beta)
                prob_theta = self.normalize_probabilities(prob_theta)
                thetas[e] = self.theta_options[np.random.choice(self.num_theta_options, p=prob_theta)]
                prob_E1 = (self.pheromones_E1[e] ** self.alpha) * (self.eta_E1 ** self.beta)
                prob_E1 = self.normalize_probabilities(prob_E1)
                E1s[e] = self.E1_options[np.random.choice(self.num_E1_options, p=prob_E1)]
                prob_density = (self.pheromones_density[e] ** self.alpha) * (self.eta_density ** self.beta)
                prob_density = self.normalize_probabilities(prob_density)
                densities[e] = self.density_options[np.random.choice(self.num_density_options, p=prob_density)]
                prob_nu12 = (self.pheromones_nu12[e] ** self.alpha) * (self.eta_nu12 ** self.beta)
                prob_nu12 = self.normalize_probabilities(prob_nu12)
                nu12s[e] = self.nu12_options[np.random.choice(self.num_nu12_options, p=prob_nu12)]
                prob_G12 = (self.pheromones_G12[e] ** self.alpha) * (self.eta_G12 ** self.beta)
                prob_G12 = self.normalize_probabilities(prob_G12)
                G12s[e] = self.G12_options[np.random.choice(self.num_G12_options, p=prob_G12)]
                prob_layup = (self.pheromones_layup[e] ** self.alpha) * (self.eta_layup ** self.beta)
                prob_layup = self.normalize_probabilities(prob_layup)
                layups[e] = self.layup_options[np.random.choice(self.num_layup_options, p=prob_layup)]
            for i in range(2):
                prob_spar = (self.pheromones_spar[i] ** self.alpha) * (self.eta_spar ** self.beta)
                prob_spar = self.normalize_probabilities(prob_spar)
                if i == 0:
                    spar_positions[i] = self.spar_position_options[np.random.choice(self.num_spar_positions, p=prob_spar)]
                else:
                    valid_indices = [j for j, pos in enumerate(self.spar_position_options)
                                   if abs(pos - spar_positions[0]) >= 1.4]
                    if not valid_indices:
                        valid_indices = list(range(self.num_spar_positions))
                    prob_spar_valid = prob_spar[valid_indices]
                    prob_spar_valid = self.normalize_probabilities(prob_spar_valid)
                    spar_positions[i] = self.spar_position_options[valid_indices[np.random.choice(len(valid_indices), p=prob_spar_valid)]]
            spar_positions.sort()
            ### Reduced random perturbation to avoid disrupting convergence (5% chance)
            if np.random.random() < 0.05:
                e = np.random.randint(self.num_elements)
                thicknesses[e] = np.random.choice(thickness_options)
                thetas[e] = np.random.choice(self.theta_options)
                E1s[e] = np.random.choice(self.E1_options)
                G12s[e] = np.random.choice(self.G12_options)
                layups[e] = np.random.choice(self.layup_options)
            if np.any(thicknesses < 1e-6):
                raise ValueError(f"Invalid solution thicknesses: {thicknesses}")
            return thicknesses, thetas, E1s, densities, nu12s, G12s, spar_positions, layups
        except Exception as e:
            logging.error(f"Solution construction failed: {e}")
            raise
    def evaluate_solution(self, thicknesses, thetas, E1s, densities, nu12s, G12s, spar_positions, layups):
        ### Evaluate a solution using FEA, using layup angles directly."""
        try:
            wing = WingGeometry(span=self.wing.span, chord=self.wing.chord,
                               num_elements_x=self.wing.num_elements_x,
                               num_elements_y=self.wing.num_elements_y,
                               spar_positions=spar_positions)
            self.fea.wing = wing
            for e in range(self.num_elements):
                self.fea.material.E1 = E1s[e]
                self.fea.material.density = densities[e]
                self.fea.material.nu12 = nu12s[e]
                self.fea.material.G12 = G12s[e]
                self.fea.material.nu21 = nu12s[e] * self.fea.material.E2 / E1s[e]
            areas = wing.get_element_areas()
            weight = np.sum(densities * thicknesses * areas)
            ### Use layup angles directly as effective thetas
            effective_thetas = layups  # 0, 45, or 90 degrees
            u, stresses, max_stress, max_displacement, freq, is_feasible = self.fea.solve(thicknesses, effective_thetas)
            stress_penalty = max(0, (max_stress / self.fea.material.sigma_y - 1)) * 50
            displacement_penalty = max(0, (max_displacement / 0.15 - 1)) * 50
            buckling_penalty = 50 if not self.fea.check_buckling(thicknesses, stresses) else 0
            penalty = stress_penalty + displacement_penalty + buckling_penalty
            total_weight = weight * (1 + penalty)
            stress_feasible = max_stress <= self.fea.material.sigma_y
            disp_feasible = max_displacement <= 0.15
            buckling_feasible = self.fea.check_buckling(thicknesses, stresses)
            logging.debug(f"Evaluated solution: weight={weight:.2f}, penalties=[stress={stress_penalty:.2f}, "
                         f"displacement={displacement_penalty:.2f}, buckling={buckling_penalty:.2f}], "
                         f"total={total_weight:.2f}, feasible=[stress={stress_feasible}, disp={disp_feasible}, "
                         f"buckling={buckling_feasible}], freq={freq:.2f} Hz")
            return total_weight, is_feasible, stress_feasible, disp_feasible, buckling_feasible, buckling_penalty, freq
        except Exception as e:
            logging.error(f"Solution evaluation failed: {e}")
            raise
    def evaluate_solutions(self, solutions):
        ### Evaluate multiple solutions sequentially
        try:
            results = []
            for i, (thicknesses, thetas, E1s, densities, nu12s, G12s, spar_positions, layups) in enumerate(solutions):
                result = self.evaluate_solution(thicknesses, thetas, E1s, densities, nu12s, G12s, spar_positions, layups)
                results.append(result)
                logging.debug(f"Evaluated solution {i+1}/{len(solutions)}")
            return results
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            raise
    def update_pheromones(self, solutions, weights, frequencies):
        ### Update pheromones based on weight
        try:
            self.pheromones_thickness *= (1 - self.rho)
            self.pheromones_theta *= (1 - self.rho)
            self.pheromones_E1 *= (1 - self.rho)
            self.pheromones_density *= (1 - self.rho)
            self.pheromones_nu12 *= (1 - self.rho)
            self.pheromones_G12 *= (1 - self.rho)
            self.pheromones_spar *= (1 - self.rho)
            self.pheromones_layup *= (1 - self.rho)
            best_weight = min(weights)
            if best_weight < float('inf'):
                best_idx = np.argmin(weights)
                best_thicknesses, best_thetas, best_E1s, best_densities, best_nu12s, best_G12s, best_spar_positions, best_layups = solutions[best_idx]
                for e in range(self.num_elements):
                    thickness = best_thicknesses[e]
                    options = self.thickness_options_spar if e in self.wing.spar_elements else self.thickness_options_skin
                    thickness_idx = np.argmin(np.abs(options - thickness))
                    self.pheromones_thickness[e, thickness_idx] += 1.0 / (best_weight + 1e-10)
                    theta = best_thetas[e]
                    theta_idx = np.argmin(np.abs(self.theta_options - theta))
                    self.pheromones_theta[e, theta_idx] += 1.0 / (best_weight + 1e-10)
                    E1 = best_E1s[e]
                    E1_idx = np.argmin(np.abs(self.E1_options - E1))
                    self.pheromones_E1[e, E1_idx] += 1.0 / (best_weight + 1e-10)
                    density = best_densities[e]
                    density_idx = np.argmin(np.abs(self.density_options - density))
                    self.pheromones_density[e, density_idx] += 1.0 / (best_weight + 1e-10)
                    nu12 = best_nu12s[e]
                    nu12_idx = np.argmin(np.abs(self.nu12_options - nu12))
                    self.pheromones_nu12[e, nu12_idx] += 1.0 / (best_weight + 1e-10)
                    G12 = best_G12s[e]
                    G12_idx = np.argmin(np.abs(self.G12_options - G12))
                    self.pheromones_G12[e, G12_idx] += 1.0 / (best_weight + 1e-10)
                    layup = best_layups[e]
                    layup_idx = np.argmin(np.abs(self.layup_options - layup))
                    self.pheromones_layup[e, layup_idx] += 1.0 / (best_weight + 1e-10)
                for i in range(2):
                    spar_pos = best_spar_positions[i]
                    spar_idx = np.argmin(np.abs(self.spar_position_options - spar_pos))
                    self.pheromones_spar[i, spar_idx] += 1.0 / (best_weight + 1e-10)
            logging.debug("Pheromones updated")
        except Exception as e:
            logging.error(f"Pheromone update failed: {e}")
            raise
    def optimize(self):
        ### Ant Colony Optimization (ACO)
        try:
            best_weight = float('inf')
            best_thicknesses = None
            best_thetas = None
            best_E1s = None
            best_densities = None
            best_nu12s = None
            best_G12s = None
            best_spar_positions = None
            best_layups = None
            start_time = time.time()
            error_count = 0
            max_errors = 10
            for iteration in range(self.max_iterations):
                iter_start = time.time()
                solutions = [self.construct_solution() for _ in range(self.num_ants)]
                try:
                    results = self.evaluate_solutions(solutions)
                except Exception as e:
                    error_count += 1
                    logging.error(f"Iteration {iteration + 1} failed: {e}")
                    if error_count >= max_errors:
                        raise RuntimeError(f"Too many errors ({error_count}) during optimization")
                    continue
                weights, feasibilities, stress_feasibilities, disp_feasibilities, buckling_feasibilities, buckling_penalties, frequencies = zip(*results)
                feasible_solutions = sum(feasibilities)
                stress_feasible = sum(stress_feasibilities)
                disp_feasible = sum(disp_feasibilities)
                buckling_feasible = sum(buckling_feasibilities)
                avg_buckling_penalty = np.mean(buckling_penalties)
                best_freq = max(frequencies)
                self.feasible_counts.append(feasible_solutions)
                self.stress_feasible_counts.append(stress_feasible)
                self.disp_feasible_counts.append(disp_feasible)
                self.buckling_feasible_counts.append(buckling_feasible)
                self.buckling_penalties.append(avg_buckling_penalty)
                self.best_frequencies.append(best_freq)
                min_weight = min(weights)
                if min_weight < best_weight:
                    best_weight = min_weight
                    best_thicknesses = solutions[np.argmin(weights)][0].copy()
                    best_thetas = solutions[np.argmin(weights)][1].copy()
                    best_E1s = solutions[np.argmin(weights)][2].copy()
                    best_densities = solutions[np.argmin(weights)][3].copy()
                    best_nu12s = solutions[np.argmin(weights)][4].copy()
                    best_G12s = solutions[np.argmin(weights)][5].copy()
                    best_spar_positions = solutions[np.argmin(weights)][6].copy()
                    best_layups = solutions[np.argmin(weights)][7].copy()
                self.best_weights.append(best_weight)
                self.update_pheromones(solutions, weights, frequencies)
                iter_time = time.time() - iter_start
                remaining_time = iter_time * (self.max_iterations - iteration - 1)
                logging.info(f"Iteration {iteration + 1}/{self.max_iterations}, "
                             f"Best Weight: {best_weight:.2f} kg, Feasible: {feasible_solutions}/{self.num_ants}, "
                             f"Stress: {stress_feasible}/{self.num_ants}, Disp: {disp_feasible}/{self.num_ants}, "
                             f"Buckling: {buckling_feasible}/{self.num_ants}, Best Freq: {best_freq:.2f} Hz, "
                             f"Iter Time: {iter_time:.2f}s, ETA: {remaining_time:.2f}s")
            total_time = time.time() - start_time
            logging.info(f"Optimization completed in {total_time:.2f}s")
            return best_thicknesses, best_thetas, best_E1s, best_densities, best_nu12s, best_G12s, best_spar_positions, best_weight
        except Exception as e:
            logging.error(f"Optimization failed: {e}")
            raise
