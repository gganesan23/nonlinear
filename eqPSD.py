# Equilibrium (non blowout) PSD

import random
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.interpolate as interpolate
from scipy import signal

# eta * Ma ^2 = 10^-3
Ma = math.sqrt(10**(-3))
K = 1
eta = 1
D = 10^-6

meandeltaT = np.pi
# variables for generating F2, H, and G1
stdF2 = eta * Ma**2
limitY2 = eta 
eqmean = 0

L = 1
M = 10**6
N = 1
m = 10**2
J = 200
dTau= 2 * np.pi / J
Tr = K * meandeltaT

y1 = 0
t = 0
initial_AT = 0
initial_dAdT = 0

def fundeltaT(M, dTau):
    n_bins = 200
    deltaT = []
    
    # generate a rayleigh distribution from which to pick values for deltaT from
    rayleigh = np.random.rayleigh(meandeltaT, M)                           
    hist, bin_edges = np.histogram(rayleigh, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    
    for i in range(M):
        # picks a random value from the rayleigh distribution
        r = np.random.rand(1)
        deltaTtemp = inv_cdf(r)
        
        # rounds deltaTtemp up or down to be an interger multiple of dTau
        if (deltaTtemp % dTau < 0.005):
            deltaTtemp2 = deltaTtemp - (deltaTtemp % dTau)
        else:
            deltaTtemp2 = deltaTtemp + dTau - (deltaTtemp % dTau)
            
        deltaT.append(deltaTtemp2[0])
    return deltaT

# generating F2, G1
def funF_k(std, limit, deltaT, eqmean, y1):
    F_k =[]
    
    for i in range(len(deltaT) - 1):
        
        # Tn is halfway through the eddy
        Tn = ((deltaT[i] + deltaT[i+1]) / 2)
        
        stdevt = ((1 - math.exp(-2 * Tn / Tr)) * std ** 2) ** (1/2)
        meant = eqmean + math.exp(-Tn / Tr) * (y1 - eqmean)
        
        # y1 is chosen from markov conditional probability function
        y1 = np.random.normal(meant, stdevt, 2)[0]
        # loops through each eddy
        for j in np.linspace(0, deltaT[i], int(deltaT[i] / dTau)):
            x = j / deltaT[i]
            if j == deltaT[i]:
                pass
            else:
                F_k.append(16 * y1 * x**2 * (1 - x)**2)
        
    return F_k

# generating F1
def fundHdt(std, limit, deltaT, eqmean, y1):
    dHdt =[]

    for i in range(len(deltaT) - 1):
        
        # Tn is halfway through the eddy
        Tn = ((deltaT[i] + deltaT[i+1]) / 2)
        
        stdevt = ((1 - math.exp(-2 * Tn / Tr)) * std ** 2) ** (1/2)
        meant = eqmean + math.exp(-Tn / Tr) * (y1 - eqmean)
        
        # y1 is chosen from markov conditional probability function
        y1 = np.random.normal(meant, stdevt, 2)[0]
        
        # loops through each eddy
        for j in np.linspace(0, deltaT[i], int(deltaT[i] / dTau)):
            x = j / deltaT[i]
            if j == deltaT[i]:
                pass
            else:
                # calculated the derivative of H = 16 * y1 * x**2 * (1 - x)**2
                dHdt.append(1 / deltaT[i] * 32 * y1 * x * (2 * x**2 - 3 * x + 1))
        
    return dHdt

def funNLAmp(F1, F2, G1, dTau, A, dAdt, N0, N1, D, t, Tstar):
    # huen's method
    # u = dA/dt

    u = dAdt
    Amp = [A]
    
    for i in range(0, int(len(F1))-2):
        m1 = u
        k1 = - ((D + F1[i]) * u ) - ( 1 + G1[i] ) * A  - (N0 + N1[i]) * A**2 + F2[i]
        m2 = u + dTau * k1
        A_2 = A + dTau * m1
        u_2 = m2
        k2 = -((D + F1[i + 1]) * u_2 ) - ( 1 + G1[i + 1] ) * A_2 - (N0 + N1[i+1]) * A_2**2 + F2[i + 1]
        m2 = u + dTau * k2
        t = t + dTau
        A = A + (dTau / 2) * (m1 + m2)
        u = u + (dTau / 2) * (k1 + k2)
        # stop when blowout occurs
        if math.isnan(A) or abs(A) > 1:
            break
        # only add to A if t > T*
        if t > Tstar:
            Amp.append(A)
    
    return Amp

def FunAvg(B):
    return sum(B)/len(B)

deltaT = fundeltaT(M, dTau)

F1 = fundHdt(stdF2, limitY2, deltaT, eqmean, y1)
F2 = funF_k(stdF2, limitY2, deltaT, eqmean, y1)
G1 = funF_k(stdF2, limitY2, deltaT, eqmean, y1)
N1 = funF_k(stdF2, limitY2, deltaT, eqmean, y1)

N0 = 1 
Tstar = 1 / (FunAvg(F1) + D)

A = funNLAmp(F1, F2, G1, dTau, initial_AT, initial_dAdT, N0, N1, D, t, Tstar)

fs = 50/(2 * np.pi)
q, PSD = signal.periodogram(A, fs)

# get rid of q = 0 so log can be taken
q = q[1:]
PSD = PSD[1:]
logPSD = np.log10(PSD)
logq = np.log10(q)

#plot
plt.plot(logq, logPSD)
plt.xlabel('log(q)', size= 14)
plt.ylabel('log(PSD)', size=15)
plt.savefig('PSDcheck.png')
