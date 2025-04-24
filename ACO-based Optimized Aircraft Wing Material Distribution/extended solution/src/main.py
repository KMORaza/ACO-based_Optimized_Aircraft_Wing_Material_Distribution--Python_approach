from wing_geometry import WingGeometry
from material_properties import Material
from fea_solver import FEASolver
from aco_optimizer import ACOOptimizer
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.sparse.linalg import eigsh

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_solution(wing, thicknesses, thetas, E1s, densities, nu12s, G12s, stresses, u, filename_prefix="wing"):
    try:
        thicknesses_mm = thicknesses * 1000
        E1s_gpa = E1s / 1e9
        densities_kgm3 = densities
        nu12s = nu12s
        G12s_gpa = G12s / 1e9
        stresses_mpa = np.array(stresses) / 1e6 if stresses is not None else np.zeros(len(wing.elements))
        nodes = wing.nodes[:, :2]
        elements = wing.elements
        quads = [nodes[element] for element in elements]
        #### Thickness Distribution
        fig, ax = plt.subplots()
        coll = PolyCollection(quads, array=thicknesses_mm, cmap='viridis', edgecolors='black')
        ax.add_collection(coll)
        for e in wing.spar_elements:
            quad = quads[e]
            ax.plot(quad[:, 0], quad[:, 1], 'r-', lw=2)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Thickness Distribution (mm)')
        plt.colorbar(coll, ax=ax, label='Thickness (mm)')
        ax.autoscale()
        plt.savefig(f"{filename_prefix}_thickness_plot.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved thickness plot to {filename_prefix}_thickness_plot.png")

        #### Fiber Angle Distribution
        fig, ax = plt.subplots()
        coll = PolyCollection(quads, array=thetas, cmap='plasma', edgecolors='black')
        ax.add_collection(coll)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Fiber Angle Distribution (degrees)')
        plt.colorbar(coll, ax=ax, label='Angle (degrees)')
        ax.autoscale()
        plt.savefig(f"{filename_prefix}_theta_plot.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved theta plot to {filename_prefix}_theta_plot.png")

        #### E1 Distribution
        fig, ax = plt.subplots()
        coll = PolyCollection(quads, array=E1s_gpa, cmap='magma', edgecolors='black')
        ax.add_collection(coll)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Young’s Modulus E1 Distribution (GPa)')
        plt.colorbar(coll, ax=ax, label='E1 (GPa)')
        ax.autoscale()
        plt.savefig(f"{filename_prefix}_E1_plot.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved E1 plot to {filename_prefix}_E1_plot.png")

        #### Density Distribution
        fig, ax = plt.subplots()
        coll = PolyCollection(quads, array=density_kgm3, cmap='cividis', edgecolors='black')
        ax.add_collection(coll)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Density Distribution (kg/m³)')
        plt.colorbar(coll, ax=ax, label='Density (kg/m³)')
        ax.autoscale()
        plt.savefig(f"{filename_prefix}_density_plot.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved density plot to {filename_prefix}_density_plot.png")

        #### Poisson's Ratio nu12 Distribution
        fig, ax = plt.subplots()
        coll = PolyCollection(quads, array=nu12s, cmap='cool', edgecolors='black')
        ax.add_collection(coll)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Poisson’s Ratio nu12 Distribution')
        plt.colorbar(coll, ax=ax, label='nu12')
        ax.autoscale()
        plt.savefig(f"{filename_prefix}_nu12_plot.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved nu12 plot to {filename_prefix}_nu12_plot.png")

        #### Shear Modulus G12 Distribution
        fig, ax = plt.subplots()
        coll = PolyCollection(quads, array=G12s_gpa, cmap='hot', edgecolors='black')
        ax.add_collection(coll)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Shear Modulus G12 Distribution (GPa)')
        plt.colorbar(coll, ax=ax, label='G12 (GPa)')
        ax.autoscale()
        plt.savefig(f"{filename_prefix}_G12_plot.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved G12 plot to {filename_prefix}_G12_plot.png")

        #### Stress Distribution
        fig, ax = plt.subplots()
        coll = PolyCollection(quads, array=stresses_mpa, cmap='inferno', edgecolors='black')
        ax.add_collection(coll)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Von Mises Stress Distribution (MPa)')
        plt.colorbar(coll, ax=ax, label='Stress (MPa)')
        ax.autoscale()
        plt.savefig(f"{filename_prefix}_stress_plot.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved stress plot to {filename_prefix}_stress_plot.png")

        #### Displacement (deformed mesh)
        fig, ax = plt.subplots()
        deformed_nodes = nodes.copy()
        for i in range(len(nodes)):
            deformed_nodes[i, 0] += u[2*i] * 100
            deformed_nodes[i, 1] += u[2*i + 1] * 100
        deformed_quads = [deformed_nodes[element] for element in elements]
        coll = PolyCollection(deformed_quads, array=thicknesses_mm, cmap='viridis', edgecolors='black')
        ax.add_collection(coll)
        ax.plot(nodes[:, 0], nodes[:, 1], 'k.', ms=5, label='Original Nodes')
        ax.plot(deformed_nodes[:, 0], deformed_nodes[:, 1], 'r.', ms=5, label='Deformed Nodes')
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Deformed Mesh (Displacement Amplified 100x)')
        plt.legend()
        ax.autoscale()
        plt.savefig(f"{filename_prefix}_displacement_plot.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved displacement plot to {filename_prefix}_displacement_plot.png")

    except Exception as e:
        logging.error(f"Failed to generate plots: {e}")

def save_solution_csv(wing, thicknesses, thetas, E1s, densities, nu12s, G12s, stresses, filename_prefix="wing"):
    try:
        np.savetxt(f"{filename_prefix}_nodes.csv", wing.nodes, delimiter=",", header="x,y,z")
        np.savetxt(f"{filename_prefix}_elements.csv", wing.elements, delimiter=",", header="n1,n2,n3,n4")
        solution_data = np.column_stack((thicknesses * 1000, thetas, E1s / 1e9, densities, nu12s, G12s / 1e9,
                                        np.array(stresses) / 1e6 if stresses is not None else np.zeros(len(wing.elements))))
        np.savetxt(f"{filename_prefix}_solution.csv", solution_data, delimiter=",",
                   header="thickness_mm,theta_deg,E1_GPa,density_kgm3,nu12,G12_GPa,stress_MPa")
        logging.info(f"Saved CSV files: {filename_prefix}_nodes.csv, {filename_prefix}_elements.csv, {filename_prefix}_solution.csv")
    except Exception as e:
        logging.error(f"Failed to save CSV files: {e}")

def plot_convergence(aco, filename="convergence_plot.png"):
    try:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(aco.best_weights) + 1), aco.best_weights, 'o-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Weight (kg)')
        ax.set_title('Convergence of Best Weight')
        ax.grid(True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved convergence plot to {filename}")
    except Exception as e:
        logging.error(f"Failed to generate convergence plot: {e}")

def plot_feasibility(aco, filename="feasibility_plot.png"):
    try:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(aco.feasible_counts) + 1), aco.feasible_counts, 'o-', label='Total Feasible')
        ax.plot(range(1, len(aco.stress_feasible_counts) + 1), aco.stress_feasible_counts, 's-', label='Stress Feasible')
        ax.plot(range(1, len(aco.disp_feasible_counts) + 1), aco.disp_feasible_counts, '^-', label='Displacement Feasible')
        ax.plot(range(1, len(aco.freq_feasible_counts) + 1), aco.freq_feasible_counts, 'd-', label='Frequency Feasible')
        ax.plot(range(1, len(aco.buckling_feasible_counts) + 1), aco.buckling_feasible_counts, 'x-', label='Buckling Feasible')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Number of Feasible Solutions')
        ax.set_title('Feasibility Over Iterations')
        ax.legend()
        ax.grid(True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved feasibility plot to {filename}")
    except Exception as e:
        logging.error(f"Failed to generate feasibility plot: {e}")

def plot_buckling_penalty(aco, filename="buckling_penalty_plot.png"):
    try:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(aco.buckling_penalties) + 1), aco.buckling_penalties, 'o-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Buckling Penalty')
        ax.set_title('Buckling Penalty Over Iterations')
        ax.grid(True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved buckling penalty plot to {filename}")
    except Exception as e:
        logging.error(f"Failed to generate buckling penalty plot: {e}")

def plot_spar_positions(wing, spar_positions, filename="spar_position_plot.png"):
    try:
        nodes = wing.nodes[:, :2]
        elements = wing.elements
        quads = [nodes[element] for element in elements]
        fig, ax = plt.subplots()
        coll = PolyCollection(quads, cmap='viridis', edgecolors='black', alpha=0.5)
        ax.add_collection(coll)
        for spar_pos in spar_positions:
            ax.axvline(x=spar_pos, color='red', linestyle='--', label=f'Spar at x={spar_pos:.1f}m')
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Spar Positions on Wing')
        ax.legend()
        ax.autoscale()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved spar position plot to {filename}")
    except Exception as e:
        logging.error(f"Failed to generate spar position plot: {e}")

def sensitivity_analysis(wing, material, fea, best_thicknesses, best_thetas, best_E1s, best_densities, best_nu12s, best_G12s, best_spar_positions):
    try:
        freqs = [0.5, 1.0, 1.5]
        weights = []
        for min_freq in freqs:
            original_check = fea.check_frequency
            def modified_check(K, M):
                eigenvalues, _ = eigsh(K, k=1, M=M, sigma=0, which='LM', tol=1e-3)
                freq = np.sqrt(np.abs(eigenvalues[0])) / (2 * np.pi)
                is_feasible = freq >= min_freq
                logging.debug(f"Sensitivity check: f={freq:.2f} Hz, min={min_freq}, feasible={is_feasible}")
                return freq, is_feasible
            fea.check_frequency = modified_check

            wing = WingGeometry(span=wing.span, chord=wing.chord,
                               num_elements_x=wing.num_elements_x,
                               num_elements_y=wing.num_elements_y,
                               spar_positions=best_spar_positions)
            fea.wing = wing
            for e in range(len(best_thicknesses)):
                fea.material.E1 = best_E1s[e]
                fea.material.density = best_densities[e]
                fea.material.nu12 = best_nu12s[e]
                fea.material.G12 = best_G12s[e]
                fea.material.nu21 = best_nu12s[e] * fea.material.E2 / best_E1s[e]
            areas = wing.get_element_areas()
            weight = np.sum(best_densities * best_thicknesses * areas)
            u, stresses, max_stress, max_displacement, freq, is_feasible = fea.solve(best_thicknesses, best_thetas)
            stress_penalty = max(0, (max_stress / fea.material.sigma_y - 1)) * 50
            displacement_penalty = max(0, (max_displacement / 0.15 - 1)) * 50
            freq_penalty = max(0, (min_freq / freq - 1)) * 50 if freq > 0 else 50
            buckling_penalty = 50 if not fea.check_buckling(best_thicknesses, stresses) else 0
            penalty = stress_penalty + displacement_penalty + freq_penalty + buckling_penalty + (0 if is_feasible else 50)
            total_weight = weight * (1 + penalty)
            weights.append(total_weight)
            logging.info(f"Sensitivity: min_freq={min_freq} Hz, weight={total_weight:.2f} kg, feasible={is_feasible}")
            fea.check_frequency = original_check
        fig, ax = plt.subplots()
        ax.plot(freqs, weights, 'o-')
        ax.set_xlabel('Minimum Frequency (Hz)')
        ax.set_ylabel('Total Weight (kg)')
        ax.set_title('Weight Sensitivity to Frequency Constraint')
        ax.grid(True)
        plt.savefig("sensitivity_plot.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info("Saved sensitivity plot to sensitivity_plot.png")
    except Exception as e:
        logging.error(f"Sensitivity analysis failed: {e}")

def main():
    try:
        wing = WingGeometry(span=10.0, chord=2.0, num_elements_x=8, num_elements_y=3)
        material = Material(E1=135e9, E2=10e9, G12=5e9, nu12=0.3, density=1600, sigma_y=1200e6)
        fea = FEASolver(wing, material)
        aco = ACOOptimizer(wing, fea, num_ants=10, max_iterations=20, rho=0.3, alpha=3.0, beta=2.0)
        best_thicknesses, best_thetas, best_E1s, best_densities, best_nu12s, best_G12s, best_spar_positions, best_weight = aco.optimize()
        wing = WingGeometry(span=10.0, chord=2.0, num_elements_x=8, num_elements_y=3,
                           spar_positions=best_spar_positions)
        fea.wing = wing
        for e in range(len(best_thicknesses)):
            fea.material.E1 = best_E1s[e]
            fea.material.density = best_densities[e]
            fea.material.nu12 = best_nu12s[e]
            fea.material.G12 = best_G12s[e]
            fea.material.nu21 = best_nu12s[e] * fea.material.E2 / best_E1s[e]
        u, stresses, max_stress, max_displacement, freq, is_feasible = fea.solve(best_thicknesses, best_thetas)
        plot_solution(wing, best_thicknesses, best_thetas, best_E1s, best_densities, best_nu12s, best_G12s, stresses, u)
        save_solution_csv(wing, best_thicknesses, best_thetas, best_E1s, best_densities, best_nu12s, best_G12s, stresses)
        plot_convergence(aco)
        plot_feasibility(aco)
        plot_buckling_penalty(aco)
        plot_spar_positions(wing, best_spar_positions)
        sensitivity_analysis(wing, material, fea, best_thicknesses, best_thetas, best_E1s, best_densities, best_nu12s, best_G12s, best_spar_positions)
        logging.info("\nOptimization Complete!")
        logging.info(f"Best Weight: {best_weight:.2f} kg")
        logging.info(f"Best Thickness Distribution: {best_thicknesses * 1000} mm")
        logging.info(f"Best Fiber Angles: {best_thetas} degrees")
        logging.info(f"Best E1 Distribution: {best_E1s / 1e9} GPa")
        logging.info(f"Best Density Distribution: {best_densities} kg/m^3")
        logging.info(f"Best Poisson’s Ratio Distribution: {best_nu12s}")
        logging.info(f"Best Shear Modulus G12 Distribution: {best_G12s / 1e9} GPa")
        logging.info(f"Best Spar Positions: {best_spar_positions} m")
        logging.info(f"Maximum Stress: {max_stress / 1e6 if stresses is not None else 'N/A'} MPa")
        logging.info(f"Maximum Displacement: {max_displacement * 1000} mm")
        logging.info(f"Lowest Frequency: {freq:.2f} Hz")
        logging.info(f"Feasible: {is_feasible}")
    except Exception as e:
        logging.error(f"Main execution failed: {e}")
        raise
if __name__ == "__main__":
    main()
