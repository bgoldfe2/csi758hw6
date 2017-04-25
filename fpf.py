# fpf.py
# JM Kinser
# 12 May 2006

import numpy as np

def FPF( data, c, fp):
    # data is the incoming data... in rows : It is converted to columns here
    # c is the constraint vector
    (N,Dim )= data.shape
    #X = transpose( data )	# put vectors into columns

    # Compute D.  Best done in rows
    D = ( np.power( abs( data), fp )).sum(0)
    #print D[0]
    D = D / N
    
    # it is possible that some values of D are 0 which will bomb out later
    ndx = (abs(D) < 0.001 ).nonzero()[0]
    D[ndx] = 0.001 * np.sign(D[ndx]+1e9)
    #print 'FPF: Zeros in D have been changed.  Number of changes =',len(ndx)

    # Y is the modified X.  Also more efficient to compute from original data
    Y = data / np.sqrt(D)
    Y = Y.transpose()
    
    # compute Q
    Yc = Y.conjugate().transpose()
    Q = Yc.dot( Y )	# inner product
    
    if N == 1:	# only 1 training vector
        Q = 1./Q
    else:
        Q = np.linalg.inv( Q )
    
    Rc = Q.dot( c)
    H = Y.dot( Rc ) / np.sqrt(D)
    # to test:  sum(conjugate(H[:,0]) * data[:,any] should equal c[any]
    return H
