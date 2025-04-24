import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WingGeometry:
    def __init__(self, span=10.0, chord=2.0, num_elements_x=10, num_elements_y=3, num_spars=2):
        self.span = span
        self.chord = chord
        self.thickness = chord * 0.05
        self.num_elements_x = num_elements_x
        self.num_elements_y = num_elements_y
        self.num_spars = num_spars
        self.element_size_x = chord / num_elements_x
        self.element_size_y = self.thickness / num_elements_y
        logging.info(f"Initializing wing geometry: span={span}, chord={chord}, elements={num_elements_x}x{num_elements_y}")
        self.nodes, self.elements, self.spar_elements = self.generate_mesh()
    def generate_mesh(self):
        ### Generate a structured quadrilateral mesh for the wing
        try:
            nodes = []
            node_counter = 0
            node_map = {}
            for i in range(self.num_elements_x + 1):
                for j in range(self.num_elements_y + 1):
                    x = i * self.element_size_x
                    y = j * self.element_size_y
                    z = 0.0
                    nodes.append([x, y, z])
                    node_map[(i, j)] = node_counter
                    node_counter += 1
            elements = []
            for i in range(self.num_elements_x):
                for j in range(self.num_elements_y):
                    n1 = node_map[(i, j)]
                    n2 = node_map[(i + 1, j)]
                    n3 = node_map[(i + 1, j + 1)]
                    n4 = node_map[(i, j + 1)]
                    elements.append([n1, n2, n3, n4])
            spar_elements = []
            if self.num_spars > 0:
                spar_positions = np.linspace(0, self.num_elements_x, self.num_spars + 2)[1:-1]
                for pos in spar_positions:
                    col = int(np.round(pos))
                    if 0 <= col < self.num_elements_x:
                        for j in range(self.num_elements_y):
                            idx = j + col * self.num_elements_y
                            if idx < len(elements):
                                spar_elements.append(idx)
            nodes = np.array(nodes)
            elements = np.array(elements)
            logging.info(f"Generated mesh: {len(nodes)} nodes, {len(elements)} elements, {len(spar_elements)} spar elements")
            return nodes, elements, spar_elements
        except Exception as e:
            logging.error(f"Mesh generation failed: {e}")
            raise
    def get_element_areas(self):
        ### Calculate the area of each element
        try:
            areas = np.full(len(self.elements), self.element_size_x * self.element_size_y)
            return areas
        except Exception as e:
            logging.error(f"Element area calculation failed: {e}")
            raise
