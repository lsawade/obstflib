# %%
# first find all directories
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

cgm = glob.glob("/autofs/nccs-svm1_home1/lsawade/gcmt/obstflib/scripts/STF_results/*/body/Z/costs_grads_main.npz")
stfdirs = [os.path.dirname(c) for c in cgm]

# %%
costs = [np.load(os.path.join(d, "costs_grads_main.npz"))['costs'] for d in stfdirs]
grads = [np.load(os.path.join(d, "costs_grads_main.npz"))['grads'] for d in stfdirs]

# %%

markersize = 3
counter = 0
durations = []
for _i, (_c, _g )in enumerate(zip(costs, grads)):
    
    
    plt.figure()
    N = len(_c)
    x = np.arange(N)*5 + 5
    if _i == 0:
        plt.plot(x, _c, 'ko', label="C", markersize=markersize)
        plt.plot(x, _g, '-o', c='tab:red', label="G", markersize=markersize)
    else:
        plt.plot(x, _c, 'ko', markersize=markersize)
        plt.plot(x, _g, '-o', c='tab:red', markersize=markersize)    
        
    plt.ylim(0.0, 0.5)

    
    plt.xlabel("Maximum source Duration T [s]")
    plt.savefig("cost_grads.pdf")
    plt.close('all')
    durfile = os.path.join(stfdirs[_i], "duration.npy")
    
    dur = ''
    if os.path.exists(durfile):
        try:
            dur = np.load(durfile)
            durations.append(dur)
        except Exception as e:
            print(e)
    
    if type(dur) == str:
        while not dur.isdigit():
            dur = input("Enter duration in seconds: ")
    
        dur = int(dur)
        np.save(durfile, dur)
        durations.append(dur)
    
    
# %%
plt.figure()
Ns = [1000, 3600, 7200]
c = ['tab:blue', 'tab:orange', 'tab:green']
x = np.arange(N)*5 + 5
for j in range(len(Ns)):
    for i in range(len(costs)):
        aic = (Ns[j] * np.log(costs[i]+1) + 2 * x)
        # normalize aic
        aic = aic - np.min(aic)
        aic = aic / np.max(aic)
        idx = np.argmin(aic)
        plt.plot(x, aic+i, 'k-', markersize=markersize)
        plt.plot(x[idx], aic[idx]+i, 'o', markersize=markersize, c = c[j])
        plt.plot
plt.savefig("CG_analysis.pdf")
    

# %%
from scipy.integrate import cumtrapz
Nstf = len(stfdirs)
N = len(costs[0])
x = np.arange(N)*5

# Normalizecumulative area under the curve
I = np.zeros((Nstf, N-1))
G = np.zeros((Nstf, N-1))
for i in range(Nstf):
    I[i, :] = cumtrapz(costs[i],x)
    G[i, :] = cumtrapz(grads[i],x)
    
I = I / I[:, -1][:, None]
G = G / G[:, -1][:, None]

# %%
# Cmopute 95% of the area
percentile = 0.99
nf = []

for i in range(Nstf):
    _idx = np.where((I[i] > percentile))[0][0]
    nf.append(x[_idx])
    
# %%

threshold_cost = 0.001
threshold_grad = 0.001

tc = []
tg = []
for i in range(Nstf):
    _idx = np.where((costs[i] < threshold_cost))[0][0]
    tc.append(x[_idx])
    
    _idx = np.where((grads[i] < threshold_grad))[0][0]
    tg.append(x[_idx])


# %%
tnf = []
tc = []
tg = []

for i in range(Nstf):
    idx = np.argmin((x[1:]-durations[i])**2)
    tnf.append(I[i, idx])
    tc.append(costs[i][idx])
    tg.append(grads[i][idx])
    
    
    
# %%

def find_elbow_point(k, curve):

    # Get number of points
    nPoints = len(curve)
    
    # Make vector with all the coordinates
    allCoord = np.vstack((k, curve)).T
    
    # First point is the first coordinate (top-left of the curve)
    firstPoint = allCoord[0]
    
    # Compute the vector pointing from first to last point
    lineVec = allCoord[-1] - allCoord[0]
    
    # Normalize the line vector
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    
    # Compute the vector from first point to all points
    vecFromFirst = allCoord - firstPoint
    
    # Compute the projection length of the projection of p onto b
    scalarProduct = np.sum(vecFromFirst * np.tile(lineVecNorm, (nPoints, 1)), axis=1)

    # Compute the vector on unit b by using the projection length
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    
    # Compute the vector from first to parallel line 
    vecToLine = vecFromFirst - vecFromFirstParallel
    
    # Compute the distance to the line
    distToLine = np.sqrt(np.sum(vecToLine**2, axis=1))
    
    # Get the index
    idxOfBestPoint = np.argmax(distToLine)
    
    return idxOfBestPoint

#%%

ebp = []
x = np.arange(N)*5
markersize = 5
i = 0

plot = False
while i < Nstf:
    
    
    Npad = 150
    _cost = np.pad(costs[i], (0, Npad), 'constant', constant_values=(0, 0))
    _grad = np.pad(grads[i], (0, Npad), 'constant', constant_values=(0, 0))
    
    _x = np.arange(N+Npad)*5 + 5
    # ic = find_elbow_point(x, costs[i]) + 1
    # ig = find_elbow_point(x, grads[i]) + 1
    ic = find_elbow_point(_x, _cost) + 1
    ig = find_elbow_point(_x, _grad) + 1
    
    # Biggest idx 
    # ebp.append(x[idx+1])
    imax = np.max([ic, ig])
    
    ebp.append(x[imax])
    
    if plot:    
        plt.figure()
        x = np.arange(N)*5 + 5
        
        plt.plot(x, costs[i], 'k-o', markersize=markersize)
        plt.plot(x, grads[i], 'r-o', markersize=markersize)
        plt.plot(x[imax], costs[i][imax], 'ko', label="All", markersize=markersize*2, markeredgecolor='k', markerfacecolor='r')
        plt.plot(x[ic], costs[i][ic], 'rx', label="Cost", markersize=markersize*2)
        plt.plot(x[ig], grads[i][ig], 'bx', label="Grad", markersize=markersize*2)
        
        plt.legend()
            
        # plt.ylim(0.0, 0.1)
        plt.xlabel("Maximum source Duration T [s]")
        plt.savefig("elbow_selection.pdf")
        plt.close('all')
        text = input(f'Current {i:d} Enter new index or - to go back or +/enter to go forward: ')
        if '-' in text:
            i=i-1
        elif '+' in text:
            i=i+1
        elif text.isdigit():
            i=int(text)  
        else:
            i += 1
    else:
        i += 1
    
    
    
# %%
# AIC



# %%

xN = np.arange(Nstf)
plt.figure()
plt.plot(xN, durations, 'k-o')
plt.plot(xN, ebp, 'b-+')
plt.xlabel("Maximum source Duration T [s]")
plt.savefig('CG_analysis.pdf')


# %%
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(xN, tnf, 'b-+')

plt.subplot(2, 1, 2)
plt.plot(xN, tc, 'k-+')
plt.plot(xN, tg, 'r-+')
plt.xlabel("Maximum source Duration T [s]")
plt.savefig('CG_analysis.pdf')

# %%
plt.figure()
plt.subplot(2, 1, 1)
plt.pcolormesh(np.arange(Nstf), x[1:], np.log10(I.T), cmap='viridis')
plt.plot(np.arange(Nstf), durations, 'k-+')
plt.plot(np.arange(Nstf), nf, 'r-+')
plt.plot(np.arange(Nstf), tc, 'b-+')
plt.plot(np.arange(Nstf), tg, 'w-+')

plt.subplot(2, 1, 2)
plt.pcolormesh(np.arange(Nstf), x[1:], np.log10(G.T), cmap='viridis')
plt.plot(np.arange(Nstf), durations, 'k-+')
plt.plot(np.arange(Nstf), nf, 'r-+')
plt.plot(np.arange(Nstf), tc, 'b-+')
plt.plot(np.arange(Nstf), tg, 'w-+')


plt.savefig('CG_analysis.pdf')

