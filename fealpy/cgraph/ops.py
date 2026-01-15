from .nodetype import CNodeType, PortConf, DataType


class Formula(CNodeType):
    """Formula Evaluation Node."""
    TITLE = "公式"
    PATH = "数据"
    DESC = """评估公式的值。
该节点支持自定义输入变量和求值表达式。
使用例子：添加输入'a'和'b'，填写代码'a+b'，则该节点执行两个变量相加。
"""
    INPUT_SLOTS = [
        PortConf("code", DataType.STRING, 0, title="公式", default=""),
        PortConf("*")
    ]
    OUTPUT_SLOTS = [
        PortConf("value", DataType.NONE, title="结果")
    ]

    @staticmethod
    def run(code, **kwargs):
        if not isinstance(code, str):
            return None
        return eval(code, kwargs)


class Script(CNodeType):
    """Script Execution Node."""
    TITLE = "脚本"
    PATH = "数据"
    DESC = """执行脚本。
该节点支持自定义输入变量和多行脚本。
使用例子：添加输入'a'和'b'和任意名称的两个输出，在“输出变量中”指定输出变量为'c, d'，填写代码'c = a + b\nd = a - b'，则该节点将返回两个变量的和与差。
"""
    INPUT_SLOTS = [
        PortConf("code", DataType.CODE, 0, title="脚本", default=""),
        PortConf("output_vars", DataType.STRING, 0, title="输出变量", default=""),
        PortConf("*")
    ]
    OUTPUT_SLOTS = [
        PortConf("*")
    ]

    @staticmethod
    def run(code, output_vars, **kwargs):
        if not isinstance(code, str):
            return None
        if not isinstance(output_vars, str):
            return None

        output_vars = output_vars.replace(",", " ").split(" ")
        output_vars = [var.strip() for var in output_vars if var != ""]
        data = dict(kwargs)
        exec(code, data)

        return tuple(data.get(var, None) for var in output_vars)
