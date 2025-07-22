
import re

def inp_reader(file_path):
    """Parse nodes and elements from an Abaqus INP file.

    This function reads the *Node and *Element sections in an Abaqus .inp file,
    extracting node IDs, coordinates, element IDs, and connectivity.

    Parameters
        file_path : str
            Path to the Abaqus INP file.

    Returns
        node_ids : list of int
            List of node IDs in the order they appear.
        node : list of list of float
            List of node coordinates, each as [x, y, z].
        cell_ids : list of int
            List of element IDs in the order they appear.
        cell : list of list of int
            List of elements, each as a list of node IDs defining its connectivity.
    """
    node_ids = []
    node = []
    cell_ids = []
    cell = []

    # compile() creates a reusable regular expression object for match() and search().
    node_pattern = re.compile(r'^\s*(\d+)\s*,\s*([\d\.\-Ee+]+)\s*,\s*([\d\.\-Ee+]+)\s*,\s*([\d\.\-Ee+]+)')
    elem_pattern = re.compile(r'^\s*(\d+)\s*,\s*(.+)')

    with open(file_path, 'r') as f:
        in_node_section = False
        in_elem_section = False

        for line in f:
            line = line.strip()

            if line.lower().startswith('*node'):
                in_node_section = True
                in_elem_section = False
                continue

            if line.lower().startswith('*element'):
                in_elem_section = True
                in_node_section = False
                continue
            
            if line.startswith('*'):
                in_node_section = False
                in_elem_section = False
                continue

            if in_node_section:
                match = node_pattern.match(line)
                if match:
                    nid, x, y, z = match.groups()
                    node_ids.append(int(nid))
                    node.append([float(x), float(y), float(z)])

            if in_elem_section:
                match = elem_pattern.match(line)
                if match:
                    cid = int(match.group(1))
                    nodes = list(map(int, match.group(2).split(',')))
                    cell_ids.append(cid)
                    cell.append(nodes)

    return node_ids, node, cell_ids, cell
