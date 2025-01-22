import scipy.optimize as opt
import numpy as np


# Bins
def compute_azimuthal_weights(az, weights=None, nbins=12, p=1):
    
    # Create the bins
    bins = np.arange(0, 360.1, 360/nbins)

    # Histogram
    H, _ = np.histogram(az, bins=bins, weights=weights)

    # Find which az is in which bin
    binass = np.digitize(az, bins) - 1

    # Compute weights
    w = (1/H[binass])**p

    # Normalize
    w /= np.mean(w)

    return w


def cosf(theta, A, C, theta0):
    return A * np.cos(theta - theta0) + C

def dcosf(theta, A, C, theta0):
    return np.vstack([np.cos(theta - theta0), np.ones_like(theta), A * np.sin(theta - theta0)])

def get_cosparams(azimuths, durations, amplitudes):
    

    def f(x, _az, yd, ya):
        [Ad, Cd, Aa, Ca, t] = x
        az = np.radians(_az)
        
        # Compute weights
        w = compute_azimuthal_weights(az, nbins=12)/len(az)
        
        # Define cost functions
        C1 = 0.5 * np.sum(w * (cosf(az, Ad, Cd, t) - yd)**2)
        C2 = 0.5 * np.sum(w * (cosf(az, Aa, Ca, t) - ya)**2)
        
        # Define gradients
        dC1 = np.sum(w * (cosf(az, Ad, Cd, t) - yd) * dcosf(az, Ad, Cd, t), axis=-1)
        dC2 = np.sum(w * (cosf(az, Aa, Ca, t) - ya) * dcosf(az, Aa, Ca, t), axis=-1)
        
        # Autonormalization using the data
        w1 = 1/np.mean(yd**2)
        w2 = 1/np.mean(ya**2)
        # print(w1 * C1, w2 *C2)
        
        return w1 * C1 + w2 *C2, np.array([*(w1 * dC1[0:2]), *(w2 * dC2[0:2]), w1 * dC1[2] + w2 * dC2[2]])
    
    # Optimization
    _opt_ = opt.minimize(f, [1, np.mean(durations), 1, np.mean(amplitudes), 0.5], args=(azimuths, durations, amplitudes), method='BFGS', jac=True)
    
    # Extract the parameters
    md = np.array([*_opt_.x[0:2], _opt_.x[4]])
    ma = np.array([*_opt_.x[2:4], _opt_.x[4]])
    
    return _opt_, md, ma



def get_cosparams_all(azimuths, starts, ends, amplitudes):
    

    def f(x, _az, ys, ye, ya):
        [As, Cs, Ae, Ce, Aa, Ca, t] = x
        az = np.radians(_az)
        
        # Compute weights
        w = compute_azimuthal_weights(az, nbins=12)/len(az)
        
        # Define cost functions
        C1 = 0.5 * np.sum(w * (cosf(az, As, Cs, t) - ys)**2)
        C2 = 0.5 * np.sum(w * (cosf(az, Ae, Ce, t) - ye)**2)
        C3 = 0.5 * np.sum(w * (cosf(az, Aa, Ca, t) - ya)**2)
        
        # Define gradients
        dC1 = np.sum(w * (cosf(az, As, Cs, t) - ys) * dcosf(az, As, Cs, t), axis=-1)
        dC2 = np.sum(w * (cosf(az, Ae, Ce, t) - ye) * dcosf(az, Ae, Ce, t), axis=-1)
        dC3 = np.sum(w * (cosf(az, Aa, Ca, t) - ya) * dcosf(az, Aa, Ca, t), axis=-1)
        
        # Autonormalization using the data
        w1 = 1/np.mean(ys**2)
        w2 = 1/np.mean(ye**2)
        w3 = 1/np.mean(ya**2)
        # print(w1 * C1, w2 *C2)
        
        return w1 * C1 + w2 *C2  + w3 *C3, np.array([*(w1 * dC1[0:2]), *(w2 * dC2[0:2]), *(w3 * dC3[0:2]), w1 * dC1[2] + w2 * dC2[2] + w3 * dC3[2]])
    
    # Optimization
    _opt_ = opt.minimize(f, [1, np.mean(starts), 1, np.mean(ends),1, np.mean(amplitudes), 0.5], args=(azimuths, starts, ends, amplitudes), method='BFGS', jac=True)
    
    # Extract the parameters
    ms = np.array([*_opt_.x[0:2], _opt_.x[6]])
    me = np.array([*_opt_.x[2:4], _opt_.x[6]])
    ma = np.array([*_opt_.x[4:6], _opt_.x[6]])
    
    return _opt_, ms, me, ma


