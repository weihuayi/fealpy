# fealpy/tests/material/test_material_library.py
import os
import pytest

from fealpy.material.material_library import MaterialLibrary
from fealpy.material.material_base import MaterialBase


@pytest.fixture
def setup_library():
    # 创建一个 MaterialLibrary 实例并返回
    library = MaterialLibrary()
    return library

def test_add_material(setup_library):
    # 测试添加材料
    library = setup_library
    steel = MaterialBase(name='Steel')
    steel.set_property('elastic_modulus', 210e9)
    library.add_material(steel)
    assert 'Steel' in library.materials
    assert library.get_material('Steel').get_property('elastic_modulus') == 210e9

def test_remove_material(setup_library):
    # 测试删除材料
    library = setup_library
    steel = MaterialBase(name='Steel')
    library.add_material(steel)
    library.remove_material('Steel')
    assert 'Steel' not in library.materials

def test_import_from_file(setup_library):
    # 测试从 JSON 文件导入材料
    library = setup_library
    test_data_path = os.path.join(os.path.dirname(__file__), '../../fealpy/material/data/metals.json')
    library.import_from_file(test_data_path, 'json')
    assert 'Steel' in library.materials

def test_export_to_file(setup_library, tmpdir):
    # 测试将材料导出到 JSON 文件
    library = setup_library
    steel = MaterialBase(name='Steel')
    steel.set_property('elastic_modulus', 210e9)
    library.add_material(steel)
    
    export_path = tmpdir.join('exported_materials.json')
    library.export_to_file(export_path, 'json')
    assert export_path.check()  # 验证文件已创建

def test_search_materials(setup_library):
    # 测试材料搜索功能
    library = setup_library
    steel = MaterialBase(name='Steel')
    library.add_material(steel)
    results = library.search_materials('Steel')
    assert 'Steel' in results

def test_load_standard_materials(setup_library):
    # 测试加载标准材料库
    library = setup_library
    library.load_standard_materials(directory='fealpy/material/data/')
    assert 'Steel' in library.materials

if __name__ == "__main__":
    test_load_standard_materials(setup_library)

