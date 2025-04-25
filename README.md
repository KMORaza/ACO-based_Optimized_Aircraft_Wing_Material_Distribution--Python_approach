## Optimizing Aircraft Wing Material Distribution using Ant Colony Optimization

* The solution is written fully in Python and it aimes to optimize the material distribution of a simplified aircraft wing cross-section to minimize its weight while satisfying structural constraints. The wing is modeled as a thin-walled rectangular section subjected to an aerodynamic lift load.
* It is essential to select the thickness and fiber orientation of composite material elements (skin and spars) to ensure the structure can withstand stress, buckling, and displacement constraints, and maintain a minimum natural frequency.
---

### Structure


1. **`material_properties.py`**:
   - Defines the `Material` class for composite material properties.
   - Computes the stiffness matrix for FEA.
2. **`wing_geometry.py`**:
   - Defines the `WingGeometry` class to generate a 2D quadrilateral mesh for the wing cross-section.
   - Manages geometric parameters and spar positions.
3. **`fea_solver.py`**:
   - Implements the `FEASolver` class for finite element analysis.
   - Computes displacements, stresses, and checks constraints.
4. **`aco_optimizer.py`**:
   - Defines the `ACOOptimizer` class for optimizing material properties, spar positions, and ply layups using ACO.
   - Evaluates solutions via FEA and updates pheromone trails.
5. **`main.py`**:
   - Orchestrates the optimization process.
   - Generates visualizations and outputs.

---

### Logic & Functionality

#### 1. `material_properties.py`
Models composite material properties and computes the stiffness matrix for FEA.

- **Class `Material`**:
  - **Initialization**:
    - Parameters: `E1` (Young’s modulus in fiber direction, 135 GPa), `E2` (transverse modulus, 10 GPa), `G12` (shear modulus, 5 GPa), `nu12` (Poisson’s ratio, 0.3), `density` (1600 kg/m³), `sigma_y` (yield stress, 1200 MPa).
    - Computes `nu21` (transverse Poisson’s ratio) as `nu21 = nu12 * E2 / E1`.
    - Logs initialization details.
  - **Method `get_stiffness_matrix(theta)`**:
    - Computes stiffness matrix `Q` for fiber angle `theta` (degrees).
    - Steps:
      1. Converts `theta` to radians.
      2. Builds compliance matrix `S` (`1/E1`, `1/E2`, `-nu12/E1`, `1/G12`).
      3. Applies transformation matrix `T` to rotate `S` to global coordinates.
      4. Inverts transformed `S` to get `Q`.
    - Includes error handling with logging.
- **Logic**:
  - The stiffness matrix relates stresses to strains, enabling FEA.
  - Fiber orientation transformation supports optimization of fiber angles.

#### 2. `wing_geometry.py`
Generates a 2D quadrilateral mesh for the wing cross-section.

- **Class `WingGeometry`**:
  - **Initialization**:
    - Parameters: `span` (10 m), `chord` (2 m), `num_elements_x` (8), `num_elements_y` (3), `spar_positions` ([0.5, 1.5] m).
    - Sets `thickness` as 5% of chord (0.1 m).
    - Computes element sizes (`element_size_x = chord / num_elements_x`, `element_size_y = thickness / num_elements_y`).
    - Calls `generate_mesh()`.
  - **Method `generate_mesh()`**:
    - Creates a grid of nodes.
    - Defines quadrilateral elements.
    - Identifies spar elements at `spar_positions`.
    - Returns nodes, elements, and spar indices.
  - **Method `get_element_areas()`**:
    - Computes element areas as `element_size_x * element_size_y` for weight calculations.
- **Logic**:
    - The mesh represents a 2D wing cross-section, with spars as structural reinforcements.
    - Optimizable spar positions enhance stiffness.
    - Error handling ensures valid mesh generation.

#### 3. `fea_solver.py`
Performs FEA to evaluate structural performance.

- **Class `FEASolver`**:
  - **Initialization**:
    - Takes `wing_geometry` and `material`.
    - Sets DOF per node (2: x, y displacements) and total DOF (`nodes * 2`).
    - Validates mesh (checks Jacobian determinants).
  - **Key Methods**:
    - **`assemble_global_stiffness(thicknesses, thetas)`**:
      - Sums element stiffness matrices (from `element_stiffness`) into global `K`.
    - **`assemble_global_mass(thicknesses)`**:
      - Builds global mass matrix `M` for frequency analysis.
    - **`element_stiffness(element, thickness, theta)`**:
      - Computes element stiffness using Gaussian quadrature, material stiffness `D`, and shape function derivatives `B`.
    - **`element_mass(element, thickness)`**:
      - Computes element mass using density and geometry.
    - **`shape_functions_and_derivatives(element, xi, eta)`**:
      - Computes shape function derivatives and Jacobian for coordinate mapping.
    - **`apply_boundary_conditions(K, F, M)`**:
      - Fixes wing root nodes (y=0).
    - **`apply_loads()`**:
      - Applies 3000 N lift to top nodes with a parabolic distribution.
    - **`check_buckling(thicknesses, stresses)`**:
      - Checks Euler buckling (spars) and plate buckling (skin).
    - **`check_frequency(K, M)`**:
      - Computes lowest natural frequency via eigenvalue analysis.
    - **`solve(thicknesses, thetas)`**:
      - Assembles `K`, `M`, applies loads and boundary conditions, solves `K u = F`, computes stresses, and checks feasibility (stress, displacement, buckling, frequency).
- **Logic**:
  - FEA evaluates wing response to loads, ensuring:
    - Stress ≤ `sigma_y`.
    - Displacement ≤ 0.15 m.
    - No buckling.
    - Adequate frequency.
  - Sparse matrices optimize computation.

#### 4. `aco_optimizer.py`
Optimizes wing design using ACO to minimize weight.


- **Class `ACOOptimizer`**:
  - **Initialization**:
    - Parameters: `wing_geometry`, `fea_solver`, `num_ants` (20), `max_iterations` (30), `rho` (0.3), `alpha` (3.0), `beta` (2.0).
    - Defines options:
      - Skin thickness: [30, 35, 40, 45] mm.
      - Spar thickness: [100, 120, 140, 160] mm.
      - Fiber angles: [0, 30, 45, 60, 90] degrees.
      - `E1`: [170, 200, 230] GPa.
      - Density: [1400, 1600, 1800] kg/m³.
      - `nu12`: [0.25, 0.3, 0.35].
      - `G12`: [8, 9, 10] GPa.
      - Spar positions: [0.2, 1.0, 1.8] m.
      - Layup angles: [0, 45, 90] degrees.
    - Initializes pheromones with biases (e.g., thicker elements, non-zero angles).
    - Sets heuristics (`eta_`) inversely proportional to magnitudes.
  - **Key Methods**:
    - **`construct_solution()`**:
      - Each ant selects values based on pheromone and heuristic probabilities.
      - Ensures spar positions are ≥1.4 m apart.
      - Adds 5% random perturbations.
    - **`evaluate_solution(...)`**:
      - Updates wing geometry and material properties.
      - Runs FEA to compute weight, stresses, displacement, frequency.
      - Applies penalties for constraint violations.
    - **`update_pheromones(solutions, weights, frequencies)`**:
      - Evaporates pheromones (`* (1 - rho)`).
      - Deposits pheromones for the best solution (`1/weight`).
    - **`optimize()`**:
      - Runs ACO:
        1. Constructs solutions.
        2. Evaluates via FEA.
        3. Updates pheromones.
        4. Tracks best design.
- **Logic**:
  - ACO uses pheromone trails to guide optimization.
  - Penalties discourage infeasible solutions.
  - Balances exploration (randomness, heuristics) and exploitation (pheromones).

#### 5. `main.py`
Integrates modules, runs optimization, and visualizes results.


- **Function `main()`**:
  - Initializes components (`WingGeometry`, `Material`, `FEASolver`, `ACOOptimizer`).
  - Runs optimization and final FEA.
  - Generates outputs.
- **Visualization Functions**:
  - **`plot_solution()`**: Plots thickness, angles, `E1`, density, `nu12`, `G12`, stress, displacement.
  - **`save_solution_csv()`**: Saves nodes, elements, solution data to CSV.
  - **`plot_convergence()`**: Plots best weight over iterations.
  - **`plot_feasibility()`**: Plots feasible solutions per constraint.
  - **`plot_buckling_penalty()`**: Plots buckling penalties.
  - **`plot_spar_positions()`**: Visualizes spar locations.
  - **`sensitivity_analysis()`**: Tests weight sensitivity to frequency constraints (0.5, 1.0, 1.5 Hz).
- **Logic**:
  - Coordinates optimization and analysis.
  - Visualizations provide design insights.
  - Sensitivity analysis explores trade-offs.

---

| ![](https://github.com/KMORaza/ACO-based_Optimized_Aircraft_Wing_Material_Distribution--Python_approach/blob/main/ACO-based%20Optimized%20Aircraft%20Wing%20Material%20Distribution/extended%20solution/convergence_plot.png) | ![](https://github.com/KMORaza/ACO-based_Optimized_Aircraft_Wing_Material_Distribution--Python_approach/blob/main/ACO-based%20Optimized%20Aircraft%20Wing%20Material%20Distribution/extended%20solution/spar_position_plot.png?raw=true)
|--|--|
| ![](https://github.com/KMORaza/ACO-based_Optimized_Aircraft_Wing_Material_Distribution--Python_approach/blob/main/ACO-based%20Optimized%20Aircraft%20Wing%20Material%20Distribution/extended%20solution/stress_histogram.png?raw=true) | ![](https://github.com/KMORaza/ACO-based_Optimized_Aircraft_Wing_Material_Distribution--Python_approach/blob/main/ACO-based%20Optimized%20Aircraft%20Wing%20Material%20Distribution/extended%20solution/wing_E1_plot.png?raw=true) |
| ![](https://github.com/KMORaza/ACO-based_Optimized_Aircraft_Wing_Material_Distribution--Python_approach/blob/main/ACO-based%20Optimized%20Aircraft%20Wing%20Material%20Distribution/extended%20solution/wing_G12_plot.png?raw=true) | ![](https://github.com/KMORaza/ACO-based_Optimized_Aircraft_Wing_Material_Distribution--Python_approach/blob/main/ACO-based%20Optimized%20Aircraft%20Wing%20Material%20Distribution/extended%20solution/wing_density_plot.png?raw=true) |
| ![](https://github.com/KMORaza/ACO-based_Optimized_Aircraft_Wing_Material_Distribution--Python_approach/blob/main/ACO-based%20Optimized%20Aircraft%20Wing%20Material%20Distribution/extended%20solution/wing_displacement_plot.png?raw=true) | ![](https://github.com/KMORaza/ACO-based_Optimized_Aircraft_Wing_Material_Distribution--Python_approach/blob/main/ACO-based%20Optimized%20Aircraft%20Wing%20Material%20Distribution/extended%20solution/wing_nu12_plot.png?raw=true) |
| ![](https://github.com/KMORaza/ACO-based_Optimized_Aircraft_Wing_Material_Distribution--Python_approach/blob/main/ACO-based%20Optimized%20Aircraft%20Wing%20Material%20Distribution/extended%20solution/wing_stress_plot.png?raw=true) | ![](https://github.com/KMORaza/ACO-based_Optimized_Aircraft_Wing_Material_Distribution--Python_approach/blob/main/ACO-based%20Optimized%20Aircraft%20Wing%20Material%20Distribution/extended%20solution/wing_theta_plot.png?raw=true) |





_Check the_ [__*webpage*__](https://optimized-aircraft-wing-design-py-aco.netlify.app/) _about the solution and its implementation_
