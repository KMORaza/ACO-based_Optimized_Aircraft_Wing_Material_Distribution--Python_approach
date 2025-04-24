from wing_geometry import WingGeometry
from material_properties import Material
from fea_solver import FEASolver
from aco_optimizer import ACOOptimizer
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_solution(wing, thicknesses, stresses, filename_prefix="wing"):
    try:
        thicknesses_mm = thicknesses * 1000  # Convert m to mm
        stresses_mpa = np.array(stresses) / 1e6 if stresses is not None else np.zeros(len(wing.elements))
        nodes = wing.nodes[:, :2]  # 2D coordinates (x, y)
        elements = wing.elements
        ### Create a list of quadrilateral vertices for each element
        quads = [nodes[element] for element in elements]
        ### Plot Thickness Distribution
        fig, ax = plt.subplots()
        coll = PolyCollection(quads, array=thicknesses_mm, cmap='viridis', edgecolors='black')
        ax.add_collection(coll)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Thickness Distribution (mm)')
        plt.colorbar(coll, ax=ax, label='Thickness (mm)')
        ax.autoscale()
        plt.savefig(f"{filename_prefix}_thickness_plot.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved thickness plot to {filename_prefix}_thickness_plot.png")
        ### Plot Stress Distribution
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

    except Exception as e:
        logging.error(f"Failed to generate plots: {e}")

def main():
    try:
        wing = WingGeometry(span=10.0, chord=2.0, num_elements_x=5, num_elements_y=2, num_spars=2)
        material = Material(E1=135e9, E2=10e9, G12=5e9, nu12=0.3, density=1600, sigma_y=1200e6)
        fea = FEASolver(wing, material)
        aco = ACOOptimizer(wing, fea, num_ants=5, max_iterations=10, rho=0.1, alpha=1.0, beta=2.0)
        best_thicknesses, best_weight = aco.optimize()
        u, stresses, max_stress, is_feasible = fea.solve(best_thicknesses)
        plot_solution(wing, best_thicknesses, stresses)
        logging.info("\nOptimization Complete!")
        logging.info(f"Best Weight: {best_weight:.2f} kg")
        logging.info(f"Best Thickness Distribution: {best_thicknesses * 1000} mm")
        logging.info(f"Maximum Stress: {max_stress / 1e6 if stresses is not None else 'N/A'} MPa")
        logging.info(f"Feasible: {is_feasible}")
    except Exception as e:
        logging.error(f"Main execution failed: {e}")
        raise
if __name__ == "__main__":
    main()
