import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ACOOptimizer:
    def __init__(self, wing_geometry, fea_solver, num_ants=5, max_iterations=10, rho=0.1, alpha=1.0, beta=2.0):
        self.wing = wing_geometry
        self.fea = fea_solver
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.num_elements = len(wing_geometry.elements)
        self.thickness_options_skin = np.array([0.005, 0.008, 0.012])  
        self.thickness_options_spar = np.array([0.015, 0.020, 0.025])  
        self.num_options = len(self.thickness_options_skin)
        self.pheromones = np.ones((self.num_elements, self.num_options)) * 0.1
        self.eta_skin = 1 / self.thickness_options_skin
        self.eta_spar = 1 / self.thickness_options_spar
        logging.info(f"Initialized ACO: {num_ants} ants, {max_iterations} iterations, {self.num_elements} elements")
    def construct_solution(self):
        ### Construct a solution (thickness distribution) for one ant
        try:
            solution = np.zeros(self.num_elements)
            for e in range(self.num_elements):
                eta = self.eta_spar if e in self.wing.spar_elements else self.eta_skin
                thickness_options = self.thickness_options_spar if e in self.wing.spar_elements else self.thickness_options_skin
                probabilities = (self.pheromones[e] ** self.alpha) * (eta ** self.beta)
                probabilities /= np.sum(probabilities) + 1e-10
                choice = np.random.choice(self.num_options, p=probabilities)
                solution[e] = thickness_options[choice]
            if np.any(solution < 1e-6):
                raise ValueError(f"Invalid solution thicknesses: {solution}")
            return solution
        except Exception as e:
            logging.error(f"Solution construction failed: {e}")
            raise
    def evaluate_solution(self, thicknesses):
        ### Evaluate a solution using FEA
        try:
            areas = self.wing.get_element_areas()
            weight = np.sum(self.fea.material.rho * thicknesses * areas)
            _, stresses, max_stress, is_feasible = self.fea.solve(thicknesses)
            stress_penalty = max(0, (max_stress / self.fea.material.sigma_y - 1)) * 200 if stresses is not None else 200
            penalty = stress_penalty + (0 if is_feasible else 100)
            total_weight = weight * (1 + penalty)
            logging.debug(f"Evaluated solution: weight={weight:.2f}, penalty={penalty:.2f}, total={total_weight:.2f}, feasible={is_feasible}")
            return total_weight, is_feasible
        except Exception as e:
            logging.error(f"Solution evaluation failed: {e}")
            raise
    def evaluate_solutions(self, solutions):
        ### Evaluate multiple solutions sequentially
        try:
            results = []
            for i, s in enumerate(solutions):
                result = self.evaluate_solution(s)
                results.append(result)
                logging.debug(f"Evaluated solution {i+1}/{len(solutions)}")
            return results
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            raise
    def update_pheromones(self, solutions, weights):
        ### Update pheromone trails
        try:
            self.pheromones *= (1 - self.rho)
            best_weight = min(weights)
            if best_weight < float('inf'):
                best_solution = solutions[np.argmin(weights)]
                for e in range(self.num_elements):
                    thickness = best_solution[e]
                    options = self.thickness_options_spar if e in self.wing.spar_elements else self.thickness_options_skin
                    option_idx = np.argmin(np.abs(options - thickness))
                    self.pheromones[e, option_idx] += 1.0 / (best_weight + 1e-10)
            logging.debug("Pheromones updated")
        except Exception as e:
            logging.error(f"Pheromone update failed: {e}")
            raise
    def optimize(self):
        ### Ant Colony Optimization
        try:
            best_weight = float('inf')
            best_thicknesses = None
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
                weights, feasibilities = zip(*results)
                feasible_solutions = sum(feasibilities)
                min_weight = min(weights)
                if min_weight < best_weight:
                    best_weight = min_weight
                    best_thicknesses = solutions[np.argmin(weights)].copy()
                self.update_pheromones(solutions, weights)
                iter_time = time.time() - iter_start
                remaining_time = iter_time * (self.max_iterations - iteration - 1)
                logging.info(f"Iteration {iteration + 1}/{self.max_iterations}, "
                             f"Best Weight: {best_weight:.2f} kg, Feasible: {feasible_solutions}/{self.num_ants}, "
                             f"Iter Time: {iter_time:.2f}s, ETA: {remaining_time:.2f}s")
            total_time = time.time() - start_time
            logging.info(f"Optimization completed in {total_time:.2f}s")
            return best_thicknesses, best_weight
        except Exception as e:
            logging.error(f"Optimization failed: {e}")
            raise
