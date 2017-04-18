
import scipy.fftpack as ft
import scipy.misc as sm
import mgcreate as mgc
import scipy.ndimage as nd
import numpy as np
import fpf

def Correlate2d( A,B ):
        a = ft.fft2( A) 
        b = ft.fft2(B)
        c = a *  b.conjugate()
        C = ft.ifft2( c )
        C = ft.fftshift( C)
        return C

def Edge( data ):
    a = data[:-1,:-1] - data[1:,1:]
    return abs(a)

def Rolls(fname):
    # load rolls, gray
    data = sm.imread( fname, flatten=True )
    # mask circle r=31
    V,H = data.shape
    mask = mgc.Circle( (V,H), (V/2,H/2), 31 )
    # correlate
    corr = Correlate2d( data, mask)
    # view
    return corr
    
def Rolls2(fname):
    # load rolls, gray
    data = sm.imread( fname, flatten=True )
    # mask circle r=31
    V,H = data.shape
    mask = mgc.Circle( (V,H), (V/2,H/2), 31 )
    # edge enhance
    a = Edge( data )
    b = Edge( mask )
    sm.imsave('dud1.png', a )
    sm.imsave('dud2.png', b)
    # correlate
    corr = Correlate2d( a,b)
    # view
    return corr
    
# circuit board
def LoadBoard( fname ):
    orig = sm.imread( fname, flatten=True)
    V,H =orig.shape
    data = mgc.Plop( orig,(2*V,2*H))
    return data

def Filter1( data ):
    rots = np.array( (0,3,6,9,12))
    N =len( rots )
    V,H = data.shape
    X = np.zeros( (N, V*H), complex )
    for i in range(len(rots)):
        r = nd.rotate( data, rots[i], reshape=False )
        R = ft.fft2( r )
        X[i] = R.ravel()
    cst = np.ones( N )
    filt = fpf.FPF( X, cst, 0.0 )
    return filt
    
def Filter2( data ):
    rots = np.array( (0,1,4,5,9))
    N =len( rots )
    V,H = data.shape
    X = np.zeros( (N, V*H), complex )
    for i in range(len(rots)):
        r = nd.rotate( data, rots[i], reshape=False )
        R = ft.fft2( r )
        X[i] = R.ravel()
    cst = np.ones( N )
    filt = fpf.FPF( X, cst, 0.0 )
    return filt 
