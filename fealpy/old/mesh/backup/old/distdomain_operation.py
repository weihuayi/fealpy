
def dmin(d0, d1):
    dd = np.concatenate((d0.reshape((-1,1)), d1.reshape((-1,1))), axis=1)
    return dd.min(axis=1)

def dmax(d0, d1):
    dd = np.concatenate((d0.reshape((-1,1)), d1.reshape((-1,1))), axis=1)
    return dd.max(axis=1)

def ddiff(d0, d1):
    return dmax(d0, -d1)


