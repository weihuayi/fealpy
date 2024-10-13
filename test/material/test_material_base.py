# fealpy/tests/material/test_material_base.py

from fealpy.material.material_base import MaterialBase


def test_material_creation():
    # 测试材料创建
    material = MaterialBase(name='TestMaterial')
    assert material.name == 'TestMaterial'
    assert material.source == 'user_defined'

def test_set_get_property():
    # 测试设置和获取材料属性
    material = MaterialBase(name='TestMaterial')
    material.set_property('elastic_modulus', 210e9)
    assert material.get_property('elastic_modulus') == 210e9

def test_calculate_property():
    # 测试通过表达式计算材料属性
    material = MaterialBase(name='TestMaterial')
    material.set_property('elastic_modulus', 210e9)
    material.set_property('poisson_ratio', 0.3)
    result = material.calculate_property('elastic_modulus * (1 - poisson_ratio)')
    assert result == 210e9 * (1 - 0.3)

def test_export_json():
    # 测试材料属性导出为 JSON 格式
    material = MaterialBase(name='TestMaterial')
    material.set_property('elastic_modulus', 210e9)
    json_data = material.export('json')
    assert '"elastic_modulus": 210000000000.0' in json_data

