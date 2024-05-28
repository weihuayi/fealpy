
import jax.numpy as jnp


class NodeSet():

    def __init__(self, NN, node=None):
        self.NN = NN
        self.nodedata['node'] = node


    def number_of_node(self):
        return self.NN 
