import random
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.interpolate as interpolate
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

axis_font = {'size':'50'}
n_bins = 200

# parameters dictating the turbulence
# eta * Ma ^2 = 10^-3
Ma = math.sqrt(10**(-3))
# volume of the mode occupied by the turbulence
K = 1
eta = 1
# eddy lifetime
meandeltaT = np.pi

# variables for stochastic turbulent functions
stdF2 = eta * Ma**2
limitY2 = eta 
eqmean = 0

# turbulence parameters
D = 10**(-6)
L = 1
M = 6*10**6
M_short = 100
N = 1
m = 10**2
J = 200
dTau= 2 * np.pi / J
Tr = K * meandeltaT

# initial conditions
y1 = 0
t = 0
initial_AT = 0
initial_dAdT = 0

# generate the lifetime of the turbulent eddies
def fundeltaT(n_bins, M, dTau):
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

# generate the stocastic turbulent functions
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

# generating the stocastic turbulent functions
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

# generate the amplitude and energy of the mode
def funNLAmp(F1, F2, G1, dTau, A, dAdt, N0, N1, D, t):
    # huen's method
    # u = dA/dt
    
    E = []
    Amp = []

    amptime = []
    u = dAdt

    # 4 terms in energy eq
    first = []
    second = []
    
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

        if math.isnan(A) or abs(A) > 1:
            break
        
        E.append(1/2 * u**2 + 1/2 * A**2 + 1/3 * N0 * A**3)
        Amp.append(A)
        amptime.append(t) 

        first.append(F2[i] * u)
        second.append(F1[i] * u**2)
    
    return Amp, amptime, E, first, second

# take time averages of components of the energy
def split(first, second, t):
    numsegments = int(t[-1]/(10**5))
    #numsegments = 10
    i = 0
    first_avg = []
    second_avg = []
    time = []
    for count in range(numsegments):
        first_segment = []
        second_segment = []
        time_segment =[]
        while (i < (count + 1) * len(first) / numsegments):
            first_segment.append(first[i])
            second_segment.append(second[i])
            time_segment.append(t[i])
            i += 1
        first_avg.append(FunAvg(first_segment))
        second_avg.append(FunAvg(second_segment))
        time.append(FunAvg(time_segment))
    return first_avg, second_avg, time

def FunAvg(B):
    return sum(B)/len(B)

# plot the energy and its components
def FunPlotEnergy(t, E, time, first, second):
    fig, axs = plt.subplots(2, 1, figsize=(40,20))
    #Remove horizontal space between axes
    fig.subplots_adjust(hspace=0) 

    axs[0].plot(t, E, 'k')
    axs[0].set_ylabel('E', **axis_font)
    axs[0].tick_params(axis = 'y', labelsize = 30)
    axs[0].set_xticks([])

    axs[1].scatter(time, first, c = 'b', marker = '.', s = 400, label = r'$F_{2}\frac{dA}{dT}$')
    axs[1].scatter(time, second, c = 'r', marker = '.', s = 400, label = r'$F_{1}(\frac{dA}{dT})^{2}$')
    axs[1].tick_params(axis = 'both', labelsize = 30)
    axs[1].set_xlabel(r'$\tau / 10^{6}$', **axis_font)
    axs[1].legend(loc = 'upper left', fontsize = 'xx-large')

    plt.savefig("fig3.png")

# plot the amplitude of the mode and its components
def FunPlotAmp(t, A, F1, F2, first, second):
    i = 0
    tmax = max(t)
    while t[i] < (t[-1] - (M_short * np.pi)):
        i += 1
    A = A[i:]
    t = t[i:]
    time = []
    for value in t:
        time.append(value - tmax)
    first = first[i:]
    second = second[i:]
    F1 = F1[i:(i + len(A))]
    F2 = F2[i:(i+len(A))]

    fig, axs = plt.subplots(5, 1, figsize=(80,50))
    # Remove horizontal space between axes
    fig.subplots_adjust(left = .2, hspace=0)  

    # Plot each graph, and manually set the y tick values
    axs[0].plot(time, A, 'k')
    axs[0].set_ylabel('A', **axis_font)
    axs[0].tick_params(axis = 'y', labelsize = 40)
    axs[0].set_xticks([])
    axs[0].set_ylim((-1,1))

    axs[1].plot(time, F2, 'k')
    axs[1].set_ylabel(r'$F_{2}$', **axis_font)
    axs[1].tick_params(axis = 'y', labelsize = 40)
    axs[1].set_xticks([])

    axs[2].plot(time, F1, 'k')
    axs[2].set_ylabel(r'$F_{1}$', **axis_font)
    axs[2].tick_params(axis = 'y', labelsize = 40)
    axs[2].set_xticks([])

    axs[3].plot(time, first, 'k')
    axs[3].set_ylabel(r'$F_{2}\frac{dA}{dT}$', **axis_font)
    axs[3].tick_params(axis = 'y', labelsize = 40)
    axs[3].set_xticks([])

    axs[4].plot(time, second, 'k')
    axs[4].set_ylabel(r'$F_{1}(\frac{dA}{dT})^{2}$', **axis_font)
    axs[4].set_xlabel(r'$\tau - \tau_{max}$', **axis_font)
    axs[4].tick_params(axis = 'both', labelsize = 40)
    axs[4].yaxis.offsetText.set_fontsize(100)

    plt.savefig("fig4.png")
    #plt.show()

# generate turbulent time scales
deltaT = fundeltaT(n_bins, M, dTau)

# generate stocastic turbulent functions
F1 = fundHdt(stdF2, limitY2, deltaT, eqmean, y1)
F2 = funF_k(stdF2, limitY2, deltaT, eqmean, y1)
G1 = funF_k(stdF2, limitY2, deltaT, eqmean, y1)
N1 = funF_k(stdF2, limitY2, deltaT, eqmean, y1)
N0 = 1

# generate the amplitude and energy of the mode
Amp, amptime, E, first, second = funNLAmp(F1, F2, G1, dTau, initial_AT, initial_dAdT, N0, N1, D, t)

# plot the amplitude, energy, and its components
FunPlotAmp(amptime, Amp, F1, F2, first, second)
first_avg, second_avg, time = split(first, second, amptime)
FunPlotEnergy(amptime, E, time, first_avg, second_avg)