# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:08:51 2019

@author: pablo
"""

import PythiaKspipiLHCb as PKp
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from timeit import default_timer as timer
N = 10000 #aprox depende de se a division é enteira ou non
processor = np.int((mp.cpu_count())*3/4) #emprego 3/4 dos recursos do nodo
seed = [i for i in range(processor)]#semilla para os distintos nucleos

def work(seed):
    test = PKp.mypythia(seed)
    p = []
    while(len(p) < (N//processor)):
        p1 = test.run(1)
        p += p1
    return p
    #P[process] += p
start = timer()
# Creo o conxunto de procesos.
pool = mp.Pool(processes = processor)
# Tomo os resultados de cada proceso, P é unha lista de p's
P = pool.map(work, seed)

pool.close()
    
pp = []       
for i in range(processor):
    pp += P[i]
#p agora debería ter os valores dos momentos dos pions:
    #p[0] = [Px,Py,Pz,p+x,p+y,p+z,p-x,p-y,p-z]
#creo o dataframe
labels = ['Px','Py','Pz','p+x','p+y','p+z','p-x','p-y','p-z']
my_pions = pd.DataFrame(pp,columns=labels)
print "time = ", timer()-start
for i in range (len(my_pions)):
    PxK = my_pions.iloc[i][0]*1000 #Paso as unidades a MeV
    PyK = my_pions.iloc[i][1]*1000
    PzK = my_pions.iloc[i][2]*1000
    Px1 = my_pions.iloc[i][3]*1000 #1 == +?
    Py1 = my_pions.iloc[i][4]*1000 
    Pz1 = my_pions.iloc[i][5]*1000
    Px2 = my_pions.iloc[i][6]*1000 #2 == -?
    Py2 = my_pions.iloc[i][7]*1000
    Pz2 = my_pions.iloc[i][8]*1000
    P1 = np.sqrt(Px1**2+Py1**2+Pz1**2)
    P2 = np.sqrt(Px2**2+Py2**2+Pz2**2)
    PK = np.sqrt((Px1+Px2)**2+(Py1+Py2)**2+(Pz1+Pz2)**2)
    cos1 = (Px1*(Px1+Px2)+Py1*(Py1+Py2)+Pz1*(Pz1+Pz2))/(P1*PK)
    cos2 = (Px2*(Px1+Px2)+Py2*(Py1+Py2)+Pz2*(Pz1+Pz2))/(P2*PK)
    PL = P1*cos1+P2*cos2
    #Take decays with momentum conservation:
    if PK == np.sqrt(PxK**2+PyK**2+PzK**2):
        my_pions.at[i,'pT'] = P1*np.sqrt(1-cos1**2)
        my_pions.at[i,'alpha'] = (P1*cos1-P2*cos2)/(P1*cos1+P2*cos2)
        my_pions.at[i,'MK'] = np.sqrt((np.sqrt(P1**2+139.57**2)+np.sqrt(P2**2+139.57**2))**2-PK**2)
        my_pions.at[i,'magPK'] = PK
    #else: my_pions.drop(my_pions.index[i])
alpha = np.asarray(my_pions['alpha'])
pt = np.asarray(my_pions['pT'])
realdata = np.zeros((len(alpha),2))
realdata[:,0] = alpha
realdata[:,1] = pt
#np.savetxt('/home3/pablo.baladron/tfm/Simulacion/Pythia_LHCb_8(2)_armenteros.txt', realdata, delimiter=',')
#
#MK = np.asarray(my_pions['MK'])
#np.savetxt('/home3/pablo.baladron/tfm/Simulacion/Pythia_LHCb_8(2)_MKhist.txt', MK, delimiter=',')
#
#magPK = np.asarray(my_pions['magPK'])
#np.savetxt('/home3/pablo.baladron/tfm/Simulacion/Pythia_LHCb_8(2)_PK.txt', magPK, delimiter=',')

plt.figure(1)
my_pions.plot(x='alpha',y='pT',markersize='0.5',style='ko',xlim=(-1,1))
plt.title=('Kspipi [Pythia]')
plt.xlabel=('alpha')
plt.ylabel=('Pt')
plt.grid()
plt.show()
#
#plt.figure(2)
#my_pions.hist(column='MK',bins=150,xlim=(497.5,497.7))
#plt.title=('MK [Pythia]')
#plt.xlabel=(r'MK (MeV)')
#plt.ylabel=('contas')
#plt.grid()
#plt.show()
