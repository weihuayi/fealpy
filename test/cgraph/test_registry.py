import unittest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from fealpy.cgraph.registry import search_all_nodes, search_node
from fealpy.cgraph.nodetype import CNodeType


class TestSearchAllNodes(unittest.TestCase):

    def setUp(self):
        """测试前准备"""
        # 清理注册表
        CNodeType.REGISTRY.clear()

    def tearDown(self):
        """测试后清理"""
        # 清理注册表
        CNodeType.REGISTRY.clear()

    @patch('fealpy.cgraph.registry.register_all_nodes')
    def test_search_all_nodes_empty_filter(self, mock_register):
        """测试 filter 为空时返回所有节点"""
        # 创建测试节点类型
        class TestNode1:
            TITLE = "Test Node 1"
            PATH = "test.path.1"
            DESC = "Test node description 1"

        class TestNode2:
            TITLE = "Test Node 2"
            PATH = "test.path.2"
            DESC = "Test node description 2"

        class TestNode3:
            # 没有 TITLE 属性，应该不可被查询
            PATH = "test.path.3"
            DESC = "Test node description 3"

        # 模拟注册表数据
        CNodeType.REGISTRY = {
            'test1': TestNode1,
            'test2': TestNode2,
            'test3': TestNode3
        }

        # 调用被测函数
        result = list(search_all_nodes())

        # 验证 register_all_nodes 被调用
        mock_register.assert_called_once()

        # 验证返回结果
        self.assertEqual(len(result), 2)

        # 验证第一个节点
        self.assertEqual(result[0]['name'], 'TestNode1')
        self.assertEqual(result[0]['title'], 'Test Node 1')
        self.assertEqual(result[0]['path'], 'test.path.1')
        self.assertEqual(result[0]['desc'], 'Test node description 1')

        # 验证第二个节点
        self.assertEqual(result[1]['name'], 'TestNode2')
        self.assertEqual(result[1]['title'], 'Test Node 2')
        self.assertEqual(result[1]['path'], 'test.path.2')
        self.assertEqual(result[1]['desc'], 'Test node description 2')

    @patch('fealpy.cgraph.registry.register_all_nodes')
    def test_search_all_nodes_with_filter(self, mock_register):
        """测试带过滤条件的搜索"""
        # 创建测试节点类型
        class NodeA:
            TITLE = "Apple Pie"
            PATH = "food.apple"
            DESC = "Apple pie recipe"

        class NodeB:
            TITLE = "Banana Bread"
            PATH = "food.banana"
            DESC = "Banana bread recipe"

        class NodeC:
            TITLE = "Apple Juice"
            PATH = "drink.apple"
            DESC = "Apple juice recipe"

        # 模拟注册表数据
        CNodeType.REGISTRY = {
            'node_a': NodeA,
            'node_b': NodeB,
            'node_c': NodeC
        }

        # 调用被测函数，过滤以 "Apple" 开头的节点
        result = list(search_all_nodes("Apple"))

        # 验证 register_all_nodes 被调用
        mock_register.assert_called_once()

        # 验证返回结果数量
        self.assertEqual(len(result), 2)

        # 验证返回的节点是正确的（Apple Pie 和 Apple Juice）
        titles = [item['title'] for item in result]
        self.assertIn('Apple Pie', titles)
        self.assertIn('Apple Juice', titles)

    @patch('fealpy.cgraph.registry.register_all_nodes')
    def test_search_all_nodes_case_insensitive(self, mock_register):
        """测试大小写不敏感搜索"""
        # 创建测试节点类型
        class TestNode:
            TITLE = "Hello World"
            PATH = "test.hello"
            DESC = "Hello world node"

        # 模拟注册表数据
        CNodeType.REGISTRY = {
            'test_node': TestNode
        }

        # 测试不同大小写的过滤条件
        result1 = list(search_all_nodes("hello"))
        result2 = list(search_all_nodes("HELLO"))
        result3 = list(search_all_nodes("Hello"))

        # 验证 register_all_nodes 被调用
        self.assertEqual(mock_register.call_count, 3)

        # 验证所有大小写变体都返回相同结果
        self.assertEqual(len(result1), 1)
        self.assertEqual(len(result2), 1)
        self.assertEqual(len(result3), 1)

        # 验证结果内容相同
        self.assertEqual(result1[0]['title'], 'Hello World')
        self.assertEqual(result2[0]['title'], 'Hello World')
        self.assertEqual(result3[0]['title'], 'Hello World')

    @patch('fealpy.cgraph.registry.register_all_nodes')
    def test_search_all_nodes_no_match(self, mock_register):
        """测试无匹配结果的情况"""
        # 创建测试节点类型
        class TestNode:
            TITLE = "Sample Node"
            PATH = "test.sample"
            DESC = "Sample node description"

        # 模拟注册表数据
        CNodeType.REGISTRY = {
            'test_node': TestNode
        }

        # 使用不匹配的过滤条件
        result = list(search_all_nodes("NonExistent"))

        # 验证 register_all_nodes 被调用
        mock_register.assert_called_once()

        # 验证返回空结果
        self.assertEqual(len(result), 0)

    @patch('fealpy.cgraph.registry.register_all_nodes')
    def test_search_all_nodes_empty_registry(self, mock_register):
        """测试注册表为空的情况"""
        # 模拟空注册表
        CNodeType.REGISTRY = {}

        # 调用被测函数
        result = list(search_all_nodes())

        # 验证 register_all_nodes 被调用
        mock_register.assert_called_once()

        # 验证返回空结果
        self.assertEqual(len(result), 0)

    @patch('fealpy.cgraph.registry.register_all_nodes')
    def test_search_all_nodes_with_none_title(self, mock_register):
        """测试 TITLE 为 None 的情况"""
        # 创建测试节点类型
        class TestNode:
            TITLE = None  # TITLE 为 None
            PATH = "test.none"
            DESC = "Node with None title"

        # 模拟注册表数据
        CNodeType.REGISTRY = {
            'test_node': TestNode
        }

        # 调用被测函数
        result = list(search_all_nodes())

        # 验证 register_all_nodes 被调用
        mock_register.assert_called_once()

        # 验证返回空结果（因为 TITLE 为 None，不满足条件）
        self.assertEqual(len(result), 0)

    @patch('fealpy.cgraph.registry.register_all_nodes')
    def test_search_all_nodes_partial_match(self, mock_register):
        """测试部分匹配的情况"""
        # 创建测试节点类型
        class Node1:
            TITLE = "Data Processing"
            PATH = "data.process"
            DESC = "Process data"

        class Node2:
            TITLE = "Data Storage"
            PATH = "data.store"
            DESC = "Store data"

        class Node3:
            TITLE = "Database Query"
            PATH = "database.query"
            DESC = "Query database"

        # 模拟注册表数据
        CNodeType.REGISTRY = {
            'node1': Node1,
            'node2': Node2,
            'node3': Node3
        }

        # 搜索包含 "Data" 的节点
        result = list(search_all_nodes("Data"))

        # 验证 register_all_nodes 被调用
        mock_register.assert_called_once()

        # 验证返回结果数量
        self.assertEqual(len(result), 3)

        # 验证返回的节点标题
        titles = [item['title'] for item in result]
        self.assertIn('Data Processing', titles)
        self.assertIn('Data Storage', titles)
        self.assertIn('Database Query', titles)


# 模拟输入槽和输出槽的数据类
@dataclass
class MockSlot:
    name: str
    type: str = ""


class TestSearchNode(unittest.TestCase):

    def setUp(self):
        """在每个测试用例之前重置 CNodeType.REGISTRY"""
        CNodeType.REGISTRY.clear()

    @patch('fealpy.cgraph.registry.register_all_nodes')  # Mock register_all_nodes 函数
    def test_search_node_exists(self, mock_register):
        """
        测试节点存在的情况：
        - 输入一个存在于 REGISTRY 中的节点名称。
        - 预期返回该节点的元信息。
        """
        # 模拟 register_all_nodes 的行为（无实际逻辑）
        mock_register.return_value = None

        # 创建一个模拟的节点类型
        mock_node_type = MagicMock()
        mock_node_type.TITLE = "TestNode"
        mock_node_type.INPUT_SLOTS = [MockSlot(name="input1", type="int"), MockSlot("*")]
        mock_node_type.OUTPUT_SLOTS = [MockSlot(name="output1", type="str")]

        # 将模拟节点注册到 REGISTRY 中
        CNodeType.REGISTRY["TestNode"] = mock_node_type

        # 调用被测函数
        result = search_node("TestNode")

        # 验证返回值
        self.assertEqual(result["title"], "TestNode")
        self.assertEqual(result["inputs"], [{"name": "input1", "type": "int"}])
        self.assertEqual(result["outputs"], [{"name": "output1", "type": "str"}])
        self.assertEqual(result["var_in"], True)
        self.assertEqual(result["var_out"], False)

    @patch('fealpy.cgraph.registry.register_all_nodes')  # Mock register_all_nodes 函数
    def test_search_node_not_exists(self, mock_register):
        """
        测试节点不存在的情况：
        - 输入一个不存在于 REGISTRY 中的节点名称。
        - 预期抛出 ValueError 异常。
        """
        # 模拟 register_all_nodes 的行为（无实际逻辑）
        mock_register.return_value = None

        # 确保 REGISTRY 为空
        CNodeType.REGISTRY.clear()

        # 调用被测函数并验证异常
        with self.assertRaises(ValueError) as context:
            search_node("NonExistentNode")

        # 验证异常信息
        self.assertIn("Node NonExistentNode not found", str(context.exception))


if __name__ == "__main__":
    unittest.main()
