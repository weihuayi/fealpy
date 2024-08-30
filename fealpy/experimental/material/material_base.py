
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

