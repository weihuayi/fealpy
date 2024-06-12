import jax.numpy as jnp

class Mesher:
    def getMeshStructure(self, mesh):
        edofMat = jnp.zeros((mesh['nelx']*mesh['nely'], 8), dtype=int)
        for elx in range(mesh['nelx']):
            for ely in range(mesh['nely']):
                el = ely + elx * mesh['nely']
                n1 = (mesh['nely'] + 1) * elx + ely
                n2 = (mesh['nely'] + 1) * (elx + 1) + ely
                edofMat = edofMat.at[el, :].set(jnp.array([2*n1+2, 2*n1+3, 2*n2+2,
                                                           2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1]))
        iK = tuple(jnp.kron(edofMat, jnp.ones((8, 1))).flatten().astype(int))
        jK = tuple(jnp.kron(edofMat, jnp.ones((1, 8))).flatten().astype(int))
        idx = (iK, jK)
        return edofMat, idx

    def getK0(self, matProp):
        E = 1.
        nu = matProp['nu']
        k = jnp.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8, \
                    -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
        KE = E / (1-nu**2) *\
        jnp.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])

        return KE
