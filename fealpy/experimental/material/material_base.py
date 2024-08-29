
class MaterialBase:
    def __init__(self, name, source='user_defined'):
        self.name = name
        self.properties = {}
        self.source = source

    def get_property(self, property_name):
        return self.properties.get(property_name, None)

    def set_property(self, property_name, value):
        self.properties[property_name] = value

    def calculate_property(self, expression):
        # 简单实现示例，使用 eval 计算表达式
        return eval(expression, {"__builtins__": None}, self.properties)

    def export(self, format='json'):
        # 导出材料参数到指定格式，支持 json、csv 等
        if format == 'json':
            import json
            return json.dumps(self.properties)
        # 其他格式处理略

class ElasticMaterial(Material):
    def __init__(self, name):
        super().__init__(name)
        self.set_property('elastic_modulus', 0)
        self.set_property('poisson_ratio', 0)

# 示例使用
steel = ElasticMaterial('Steel')
steel.set_property('elastic_modulus', 210e9)
steel.set_property('poisson_ratio', 0.3)
print(steel.get_property('elastic_modulus'))
print(steel.calculate_property('elastic_modulus * (1 - poisson_ratio)'))

