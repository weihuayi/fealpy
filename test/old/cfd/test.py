#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: test.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2023年12月06日 星期三 19时37分46秒
	@bref 
	@ref 
'''  
import numpy as np
import pytest

class MyClass:
    def __init__(self, value):
        self.value = value
    
    def fun(self,a):
        return a
# 创建一个 fixture，用来生成预先的实例
@pytest.fixture
def my_instance():
    instance = MyClass(100)  # 创建一个实例
    return instance

# 测试函数1，使用 my_instance fixture
def test_instance_value(my_instance):
    c = my_instance.fun(2)
    assert my_instance.value == 100

# 测试函数2，也使用 my_instance fixture
def test_instance_type(my_instance):
    assert isinstance(my_instance, MyClass)

