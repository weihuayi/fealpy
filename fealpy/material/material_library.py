import json
import csv

from .material_base import MaterialBase

class MaterialLibrary:
    def __init__(self):
        self.materials = {}  # 存储所有材料对象
        self.standard_sources = ['MATWEB', 'ANSYS', 'COMSOL']  # 可扩展的标准来源

    def add_material(self, material):
        """添加材料到材料库"""
        if material.name in self.materials:
            print(f"Material {material.name} already exists. Updating its data.")
        self.materials[material.name] = material

    def remove_material(self, material_name):
        """删除指定名称的材料"""
        if material_name in self.materials:
            del self.materials[material_name]
            print(f"Material {material_name} removed from the library.")
        else:
            print(f"Material {material_name} not found in the library.")

    def get_material(self, material_name):
        """获取指定名称的材料对象"""
        return self.materials.get(material_name, None)

    def import_from_file(self, filepath, filetype='json'):
        """从文件导入材料数据，支持 JSON 和 CSV 格式"""
        if filetype == 'json':
            with open(filepath, 'r') as file:
                data = json.load(file)
                for name, properties in data.items():
                    material = MaterialBase(name)
                    material.properties = properties
                    self.add_material(material)
        elif filetype == 'csv':
            with open(filepath, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    material = MaterialBase(row['name'])
                    # 假设 CSV 文件的其他列为属性
                    material.properties = {k: float(v) for k, v in row.items() if k != 'name'}
                    self.add_material(material)
        else:
            print(f"Unsupported file type: {filetype}")

    def export_to_file(self, filepath, filetype='json'):
        """导出材料数据到文件，支持 JSON 和 CSV 格式"""
        if filetype == 'json':
            with open(filepath, 'w') as file:
                data = {name: material.properties for name, material in self.materials.items()}
                json.dump(data, file, indent=4)
        elif filetype == 'csv':
            if not self.materials:
                print("No materials to export.")
                return
            with open(filepath, 'w', newline='') as file:
                fieldnames = ['name'] + list(next(iter(self.materials.values())).properties.keys())
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for name, material in self.materials.items():
                    row = {'name': name}
                    row.update(material.properties)
                    writer.writerow(row)
        else:
            print(f"Unsupported file type: {filetype}")

    def search_materials(self, keyword):
        """根据关键词搜索材料"""
        results = [name for name in self.materials if keyword.lower() in name.lower()]
        if results:
            print(f"Found materials: {', '.join(results)}")
        else:
            print("No materials found matching the keyword.")
        return results

    def list_materials(self):
        """列出库中所有材料名称"""
        if not self.materials:
            print("No materials in the library.")
        else:
            print("Materials in the library:")
            for name in self.materials:
                print(f"- {name}")

    def load_standard_materials(self, directory='./data/'):
        """加载标准材料库中的所有材料数据"""
        import os
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                self.import_from_file(os.path.join(directory, filename), 'json')
