# -*- coding: utf-8 -*-
# HW5 baseball.py
"""Created on Wed Apr 12, 2017

@author: bgoldfeder
"""

import os,sys,cv2
import numpy as np
import scipy.misc as sm
import scipy.fftpack as ft
import mgcreate as mg
import fpf

def LoadTach(fname):
    img = sm.imread(fname,flatten=True)
    V,H = img.shape
    X = np.zeros( (9,V*H), complex )
    cst = np.array([1,-1,1,-1,1,-1,1,-1,1])

    #sm.imsave("tach1.png",img)
    # Read in the nine cutout images of the numbers and make FPF
    for i in range(0,9):
        iname = str(i)+".png"
        oname = str(i) + "b.png"
        numb = BandW(iname,oname)
        out = mg.Plop(numb,(V,H))
        #sm.imsave("numbers\\" + oname,out)
        X[i] = ft.fft2( out ).ravel()

    filt = fpf.FPF( X, cst, 0)
    filt = ft.ifft2( filt.reshape((V,H)))
    #sm.imsave("filt.png",filt.real)
    return img,filt

def FudgeFilt():
    V = 446
    H = 432
    X = np.zeros( (9,V*H), complex )
    cst = np.array([1,-1,1,-1,1,-1,1,-1,1])
    for i in range(0,9):
        iname = "num2/"+str(i)+"b.png"
        out = sm.imread(iname,flatten=True)
        X[i] = ft.fft2( out ).ravel()
    filt = fpf.FPF( X, cst, 0)
    filt = ft.ifft2( filt.reshape((V,H)))
    #sm.imsave("filt.png",filt.real)
    return filt

def FindNumbs(img,filt):
    a = Edge(img)
    b = Edge(filt)
    corr = Correlate2d(a,b)
    return corr    

def Correlate2d( A,B ):
        a = ft.fft2( A) 
        b = ft.fft2(B)
        c = a *  b.conjugate()
        C = ft.ifft2( c )
        C = ft.fftshift( C)
        return C
    

def BandW(fname,outname):
    img = cv2.imread(fname,0)    
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)    
    #cv2.imwrite(outname,thresh1)    
    return thresh1

def Edge( data ):
    a = data[:-1,:-1] - data[1:,1:]
    return abs(a)

