
import numpy as np
from matplotlib import pyplot as plt


def getpolicy(Q):
    """ Get best policy matrix from the Q-matrix.
    You have to implement this function yourself. It is not necessary to loop
    in order to do this, and looping will be much slower than using matrix
    operations. It's possible to implement this in one line of code.
    """

    P = np.argmax(Q, axis=2)

    return P


def getvalue(Q):
    """ Get best value matrix from the Q-matrix.
    You have to implement this function yourself. It is not necessary to loop
    in order to do this, and looping will be much slower than using matrix
    operations. It's possible to implement this in one line of code.
    """
    
    V = np.max(Q, axis=2)

    return V

