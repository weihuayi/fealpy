
from typing import Type, Optional, Any
from ..backend import bm

class MeshData(dict):
    """
    """
    def update_node_id(self, cell):
        """
        Update the node ids in the cell data to match the current node ids.
        """
        nids = self.get_node_data('id')
        N = bm.max(nids) + 1
        idmap = bm.zeros(N, dtype=bm.int32)
        idmap = bm.set_at(idmap, nids, bm.arange(len(nids), dtype=bm.int32))
        cell = idmap[cell]
        # Update node ids in the node sets
        for name, data in self.get("node_sets", {}).items():
            self.add_node_set(name, idmap[data])

        self['nidmap'] = idmap
        return cell

    def get_rbe2_edge(self): 
        """
        """
        rbe2 = []
        rnodes = []
        for name, data in self.get('couplings', {}).items():
            surface = self.get_surface(data['surface'])
            nset = self.get_node_set(surface['entity_set'][0][0])
            rnode = self.get_node_set(data['ref_node'])
            edge = bm.stack([bm.full_like(nset, rnode[0]), nset], axis=1)
            rbe2.append(edge)
            rnodes.append(rnode[0])

        return bm.concat(rbe2, axis=0), bm.array(rnodes, dtype=bm.int32)

        
    def add_node_data(self, name, data):
        self.setdefault("node_datas", {})[name] = data

    def get_node_data(self, name):
        return self.get("node_datas", {}).get(name, None)

    def add_cell_data(self, name:str, data):
        self.setdefault("cell_datas", {})[name] = data

    def get_cell_data(self, name: str):
        return self.get("cell_datas", {}).get(name, None)

    def add_node_set(self, name, nids):
        self.setdefault("node_sets", {})[name] = nids 

    def get_node_set(self, name):
        return self.get("node_sets", {}).get(name, None)

    def add_cell_set(self, name, cids):
        self.setdefault("cell_sets", {})[name] = cids 

    def get_cell_set(self, name):
        return self.get("cell_sets", {}).get(name, None)

    def add_material(self, name, **kwargs):
        self.setdefault("materials", {})[name] = kwargs

    def get_material(self, name):
        return self.get("materials", {}).get(name, None)


    def add_surface(self, 
                    name: str, 
                    surface_type: str, 
                    entity_set: list[tuple[str, str]]):
        self.setdefault("surfaces", {})[name] = {
                "type": surface_type,
                "entity_set": entity_set,
                }

    def get_surface(self, name: str):
        return self.get("surfaces", {}).get(name, None)


    def add_coupling(self, 
                     name: str, 
                     surface: str, 
                     ref_node: str, 
                     coupling_type: str):
        self.setdefault("couplings", {})[name] = {
                "surface": surface,
                "ref_node": ref_node,
                "type": coupling_type,
                }

    def get_coupling(self, name: str):
        return self.get("couplings", {}).get(name, None)


    def add_physics_resions(self, 
                            name: str, 
                            cell_set: str, 
                            material: str):
        self.setdefault("physics_regions", {})[name] = {
                "cell_set": cell_set,
                "material": material,
                }

    def get_physics_region(self, name: str):
        return self.get("physics_regions", {}).get(name, None)


    def add_boundary_condition(self, bc):
        self.setdefault("boundary_conditions", []).append(bc)

    def get_boundary_conditions(self):
        return self.get("boundary_conditions", [])
