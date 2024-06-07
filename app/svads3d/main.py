import os
import numpy as np
import json
from typing import Union


if __name__ == '__main__':
    file_path = './data.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    print(data)
    pass
