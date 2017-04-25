# fftdock.py2
import numpy as np
import scipy.ndimage as nd
import mgcreate as mg
import gogabor as ggb
from scipy.ndimage.interpolation import shift
import PIL as pil
import fpf
import scipy.misc as sm
import scipy.fftpack as ft

def BuildDock():
    V=148
    H=543
    dock = np.zeros( (V,H) ) # frame size
    
    # long pier
    dock[68:77, 32:543] = 1
    # extra dock vertical
    dock[31:69,41:47] = 1
    
    #repeating top dock verticals
    for i in range( 14 ):
        x = 61 + (i * 35.8) + 0 
        dock[30:68, x:x+6] = 1
   
    # extra dock bottom vertical
    dock[77:116,34:40] = 1

    #repeating bottom dock verticals
    for i in range( 14 ):
        x = 61 + (i * 35.8)+0
        dock[76:119,x:x+6] = 1
        
    return dock

def LocateDock(origot,dock):
    corr = ggb.Correlate2d(origot,dock)
    V,H = corr.shape
    v,h = divmod( abs(corr).argmax(), H )
    return v,h

def Overlay( origrot, dock, vh ):
    v,h = vh
    V,H = dock.shape
    ndock = shift( dock, (v-V/2, h-H/2),mode='wrap' ) # causes non zero values to appear
    ndock = ndock > 0.01
    return origrot + 300*ndock

def SubtractDock( origrot, dock, vh ):
    v,h = vh
    V,H = dock.shape
    bigdock = nd.binary_dilation( dock, iterations=2)
    ndock = shift( bigdock, (v-V/2, h-H/2) )
    ndock = ndock > 0.1
    return origrot -200*ndock

def SubtractDock2(origrot, dock, vh):
    newimg = origrot * dock
    return newimg

def Roll(image, delta):
    "Roll an image sideways"
    v, h = image.shape
    blank = np.ones((v,h))
    for i in range(h):
        for j in range(v-2):
            if image[j,i] == 0:
                #not doing y or wrap yet
                blank[j+delta[0],i] = 0
                #print(i,j,"one!")
    
    return blank

def MakeFPF(numBoat,cst):
    V=148
    H=543
    X = np.zeros( (numBoat,V*H), complex )
    #cst = np.array([1,1])
    for i in range(0,numBoat):
        iname = "boat"+str(i+1)+"bwFilt.png"
        print("read file ",iname)
        out = sm.imread(iname,flatten=True)
        X[i] = ft.fft2( out ).ravel()

    print("X shape",X.shape)
    filt = fpf.FPF( X, cst, 1)
    print("X shape",X.shape)
    filt = ft.ifft2( filt.reshape((V,H)))
    #sm.imsave("filt.png",filt.real)
    return filt

def MakeBoatCutOut(fname,V,H):
    # Utility for taking b&w boat cutouts into full size filters
    cut = sm.imread(fname,flatten=True)
    filtCut = mg.Plop(cut,(V,H))
    return filtCut

def FindBoats(img,filt):
    #a = Edge(img)
    #b = Edge(filt)
    #corr = Correlate2d(a,b)
    corr = Correlate2d(img,filt)
    return corr    

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

def Driver():
    origrot = sm.imread("boat2a.png",flatten=True)
    dock = BuildDock()
    v,h = LocateDock(origrot,dock)
    print("Dock center location",v,",",h)
    overlaydock = Overlay(origrot,dock,(v,h))
    sm.imsave("Overlay.png",overlaydock)

    # I am not getting good results with the correlation
    # I think Shift is the issue.
    
    imgNoDock = SubtractDock(origrot, dock,(v,h))
    sm.imsave("imgNoDock1.png",imgNoDock)

    # Cheating a little bit I subtract out by using the binary opposite
    # of the dock mask using the correlation adjustment against the
    # Center of Mass
    dockOpp = (dock - 1) * -1
    sm.imsave("dockopp.png",dockOpp)
    com = nd.measurements.center_of_mass(dockOpp)
    print("center of mass",com)
    delta = (int(v-com[0]),int(h-com[1]))
    print("delta is",delta)
    corr = Roll(dockOpp,delta)
    sm.imsave("Corr2.png",corr)

    # Now use subtract2 instead
    imgNoDock2 = SubtractDock2(origrot,corr,(v,h))
    sm.imsave("nodock2.png",imgNoDock2)

    # Create FPF providing the number of filter images and the constraint matrix
    filtB1 = MakeFPF(4,np.array([1,1,1,-1]))
    corrB1 = FindBoats(imgNoDock2,filtB1)
    sm.imsave("filtB1Ans.png",corrB1.real)
    
    
    

    
