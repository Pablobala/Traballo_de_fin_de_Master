# -*- coding: utf-8 -*-
import pythia8
import numpy as np
#primeiro paso, engadir ao teu pythonpath pythia
#setenv PYTHONPATH ${PYTHONPATH}:/scratch02/xabier.cid//pythia8235/lib/

## find all the descendants, if i0 then save the PID family tree
def descendants(ev,par,myl,i0=-1):
    par0 = par
    if type(par)==list: par0 = par[1]
    for da in par0.daughterList():
        if i0>=0:
            myl.append([i0,ev[da].index(),ev[da].name()])
            descendants(ev,ev[da],myl,i0+1)
        else:
            myl.append(da)
            descendants(ev,ev[da],myl)
    return

#take from ks decays to p Kspipi events 
def Kspipi(ev,ks0, p):
#asumimos que se hai un estado final con pion cargado vamolo coller, pode haber un pi0 (CP) ou gamma
    if type(ks0)==list: M = ks0[1]
    else: M = ks0
    #podo intentar eliminar os eventos con gammas pa minimizar a dispersion da elipse
    if len (M.daughterList()) == 2:
        m1 = [x for x in M.daughterList() if ev[x].id() == 211]
        m2 = [x for x in M.daughterList() if ev[x].id() == -211]
        if len(m1) == 0 or len(m2) == 0: return
        else:
            m1 = m1[0]
            m2 = m2[0]
            if ev[m1].zProd() == ev[m2].zProd() == M.zDec():
                if 5. < np.sqrt(M.px()**2+M.py()**2+M.pz()**2) < 200.: #momento optimo para LHCb
                    if 2. < ev[m1].eta() < 5. and 2. < ev[m2].eta() < 5.:
                        p.append([M.px(),M.py(),M.pz(),ev[m1].px(),ev[m1].py(),ev[m1].pz(),ev[m2].px(),ev[m2].py(),ev[m2].pz()])
    return

## has the particle flown?
def isprompt(par):
   xdec,xprod = par.xDec(),par.xProd()
   ydec,yprod = par.yDec(),par.yProd()
   zdec,zprod = par.zDec(),par.zProd()
   return (xdec==xprod and ydec==yprod and zdec==zprod)



class mypythia:
    
    def __init__(self,jind=0):

        self.py = pythia8.Pythia("", False)
        self.py.readString("SoftQCD:nonDiffractive = on")
        self.py.readString("Beams:eCM = 13000")
        self.py.readString("Print:quiet = on")
        self.py.readString("Random:setSeed = on")
        self.py.readString("Random:seed = "+str(jind))
        self.py.init()

    def run(self,nevts=10000):
        for j in xrange(nevts):
            if j and (not j%10000):
                print "*********\n",j,"events gone\n*********"
            if not self.py.next(): continue
            ev = self.py.event

            #http://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
            ## estes son os KS0s
            ks0s = [x for x in ev if abs(x.id())==310]
            p = []
            for ks in ks0s:
                Kspipi(ev,ks,p)
                #faltaríame coller todos os nevts en vez de so o ultimo
                #agora teño que facer x.run(1)
                #print ks.id()        
        return  p
