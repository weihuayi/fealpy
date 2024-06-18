#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: tgv.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Thu 13 Jun 2024 03:18:04 PM CST
	@bref 
	@ref 
'''  
import numpy as np
from fealpy.jax.sph import NodeMesh

mesh = NodeMesh.from_tgv_domain()
