# -*- coding: utf-8 -*-
"""
Created on Fri May  3 19:01:46 2019

@author: pablo
"""

from tools import initialize
initialize(1)
import pycuda.gpuarray as gpuarray
import numpy as np
import toyrand
import pycuda
import pycuda.curandom
from ModelBricks import cuRead
from scipy import random as rnd
from matplotlib import pyplot as plt

BLOCKSIZE = 128
mod = cuRead("theil2.c")
Theil = mod.get_function("Theil")
nps = np.int64(BLOCKSIZE*3000) #10000# numero de hipoteses pras masas K, pi . Fagoi multiplo do BLOCKSIZE por conveniencia
#ngi = 100 ## number of toys per hypothesis, to improve precision in the average
bins = 150 ## si o cambias, hai q cambialo tamen no theil.c
y_binnum= np.int32(bins) #number of bins along the y-axis
x_binnum= np.int32(bins) #number of bins along the x-axis
    
phi_min = -1.5708
phi_max = 1.5708
r_min = 0.999992
r_max = 1.000006
MPDG = 497.61
mPDG = 139.57
phis = np.arange(phi_min, phi_max, (phi_max - phi_min)*1./x_binnum)
rs = np.arange(r_min, r_max, (r_max - r_min)*1./y_binnum)

momentums = np.loadtxt("Pythia_statistics_P.txt",delimiter=',') #MeV
momentums = np.asarray(momentums)
momentums = momentums[:40000] #Nº de datos que emprego 
scales = [1., .999999, 1.000001, .999998, 1.000002, .999997, 1.000003, .999996, 1.000004, .999995, 1.000005]

for scale in scales:
    momentums1 = momentums*1000*scale #En MeV
    realdata = np.zeros((len(momentums1),2))
    for i in range (len(momentums1)):
        Px1 = momentums1[i,0] #1 == +?
        Py1 = momentums1[i,1] 
        Pz1 = momentums1[i,2]
        Px2 = momentums1[i,3] #2 == -?
        Py2 = momentums1[i,4]
        Pz2 = momentums1[i,5]
        P1 = np.sqrt(Px1**2+Py1**2+Pz1**2)
        P2 = np.sqrt(Px2**2+Py2**2+Pz2**2)
        PK = np.sqrt((Px1+Px2)**2+(Py1+Py2)**2+(Pz1+Pz2)**2)
        cos1 = (Px1*(Px1+Px2)+Py1*(Py1+Py2)+Pz1*(Pz1+Pz2))/(P1*PK)
        cos2 = (Px2*(Px1+Px2)+Py2*(Py1+Py2)+Pz2*(Pz1+Pz2))/(P2*PK)
        PL = P1*cos1+P2*cos2
        realdata[i,1] = P1*np.sqrt(1-cos1**2)
        realdata[i,0] = (P1*cos1-P2*cos2)/(P1*cos1+P2*cos2)
    
    #realdata = np.asarray(realdata)
    
    def elliptical(M,m,x,y): #defino as coordenadas elipticas
        a = np.sqrt(M**2-4.*m**2)/M #o propio plano ten uns parámetros a e b
        b = np.sqrt(M**2/4.-m**2)
        phi = np.arctan(y*a/(x*b))
        r = np.sqrt(x**2/a**2+y**2/b**2)    
        return phi, r
    def gpu_toy(M,m,Mt,mt,x):
        one = np.ones(nps)
        #ones = pycuda.gpuarray.to_gpu(one)
        a = np.sqrt(M**2-4.*m**2)/M
        b = np.sqrt(M**2/4.-m**2)
        at = (pycuda.cumath.sqrt(Mt**2-4.*mt**2)/Mt).get()
        bt = (pycuda.cumath.sqrt(Mt**2/4.-mt**2)).get()
        rt = np.sqrt(one/((a/at*np.cos(x))**2+(b/bt*np.sin(x))**2))
        return rt
    
    y_bin = rs
    x_bin = phis
    def fill(xary, yary):
        StartMatrix = bins*bins*[0.]
        for k in xrange(len(xary)):
            x = xary[k]
            y = yary[k]
            donex, doney = False, False
            w =.5*( y_bin[1]-y_bin[0]);
            for j in xrange(bins): 
                ymax = y_bin[j] + w ;
                ymin = y_bin[j] - w;
                
                if ymax > y and ymin < y :
                    #print "Here"
                    doney = True
                    break
            if not doney: continue
            w =.5*( x_bin[1]-x_bin[0]);
            for i in xrange(bins): 
                xmax = x_bin[i] + w ;
                xmin = x_bin[i] - w;
                if xmax > x and xmin < x :
                    #print "There"
                    donex = True
                    break
            if not donex:continue
            StartMatrix[bins*i + j] += 1
        StartMatrix = np.float64(StartMatrix)        
        return StartMatrix
    realdataphi, realdatar = elliptical(MPDG,mPDG,realdata[:,0],realdata[:,1])
    StartMatrix = fill(realdataphi,realdatar)
    
    kaon_gpu =  (497.611-497.609)*pycuda.curandom.rand(nps, dtype = np.float64) + np.float64(497.609)
    pion_gpu = (139.571-139.569)*pycuda.curandom.rand(nps, dtype = np.float64) + np.float64(139.569)
    
    indices = gpuarray.to_gpu(np.float64(nps*[0.]))
    pts = []
    for phi in phis: pts.append(gpu_toy(MPDG,mPDG,kaon_gpu,pion_gpu,phi))
    
    pts = np.float64(pts)
    
    data_gpu = gpuarray.to_gpu(StartMatrix)
    Nevts = np.int64(gpuarray.sum(data_gpu).get())
    
    pts_gpu = gpuarray.to_gpu(np.float64(pts))
    pt_ax_gpu = gpuarray.to_gpu(np.float64(rs))
    
    Theil(nps, indices, data_gpu, pt_ax_gpu, pts_gpu,
          Nevts, 
          block = (BLOCKSIZE,1,1), grid = (nps/BLOCKSIZE,1,1))
    
    theils = indices.get()
    pion = pion_gpu.get()
    kaon = kaon_gpu.get()
    all_shit = []
    
    best_th = gpuarray.max(indices).get()
    mask = indices == best_th
    S = gpuarray.sum(mask).get()
    mpi = gpuarray.sum(pion_gpu*mask)/S
    mk = gpuarray.sum(kaon_gpu*mask)/S
    
    if S > 1: print "interesting, " , S, " points out of ", len(mask), "have all the maximum Theil index, will average them"
    print "Best Theil:", best_th, "Mpi: ", mpi, "MK: ", mk, "Scale: ", scale
    print "##############################################"
    result = [best_th, mpi.get(), mk.get()]
    
    
    #ploteo o resultado neste espazo:
    def toy(x, Mt,mt,M=MPDG,m=mPDG):
        a = np.sqrt(M**2-4.*m**2)/M
        b = np.sqrt(M**2/4.-m**2)
        at = np.sqrt(Mt**2-4.*mt**2)/Mt
        bt = np.sqrt(Mt**2/4.-mt**2)
        rt = np.sqrt(np.ones(np.shape(x))/((a/at*np.cos(x))**2+(b/bt*np.sin(x))**2))
        return rt
      
    xx=np.linspace(phi_min,phi_max,800)
    plt.figure(1)
    plt.plot(realdataphi,realdatar,linestyle='None',markersize='0.1',marker='o',color='g',label='MKmpiPDG-DecayByHand')
    #plt.plot(xx,toy(xx,all_shit[-1][2],all_shit[-1][1]),'r-',label='Axuste gpu')
    plt.plot(xx,toy(xx,result[2],result[1]),'r-',label='Axuste gpu')
    plt.legend(loc='best')
    plt.title(r'Armenteros-Podolanski (elliptical coordinates) plot')
    plt.xlabel(r'$\phi$')
    plt.ylabel('r')
    plt.grid()
    plt.show()
    
