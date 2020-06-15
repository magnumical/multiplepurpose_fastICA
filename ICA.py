# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pyedflib


#%% Read EEG
eegFile = pyedflib.EdfReader('edfdata.edf')
signal_labels = eegFile.getSignalLabels()
annotation = eegFile.readAnnotations()

fs=160

lenght=annotation[1][0]*fs

trial1 = np.zeros((64, 672))
for i in np.arange(64):
    trial1[i, :] = eegFile.readSignal(i)[:672]
    
xaxis=list(range(0, 672))

m, n = trial1.shape

#%% Plot the trial!
plt.figure()
#c4
plt.plot(xaxis, trial1[12])

plt.xlabel('4.2 sec - trial 1')
plt.ylabel('EEG')
plt.title('PhysioNet EEG Motor Dataset ')
plt.show()


#%%whitening
mean =np.mean(trial1, axis=1, keepdims=True)

trial1_minus_mean =trial1 - mean

covariance=(trial1_minus_mean.dot(trial1_minus_mean.T))/(np.shape(mean)[0] - 1)

# Single value decoposition: see its documentation!
U, S,Vh = np.linalg.svd(covariance)

#eigenvalues : diagonal matrix
eigenvalues = np.diag(1.0 / np.sqrt(S)) 

#whitening matrix
whiteM = np.dot(U, np.dot(eigenvalues, U.T))

# apply whitening matrix
Whitened_sig = np.dot(whiteM, trial1_minus_mean) 


#%%

# create random weights
W = np.random.rand(m, m)
alpha = .5
thresh=1e-10
iterations=1e10

for c in range(m):
        w = W[c, :].copy().reshape(m, 1)
        w = w / np.sqrt((w ** 2).sum())

        i = 0
        lim = 100
        while ((lim > thresh) & (i < iterations)):

            # Dot product of weight and signal
            ws = np.dot(w.T, trial1)

            # Pass w*s into contrast function g
            wg = np.tanh(ws * alpha).T

            # Pass w*s into g prime 
            wg_ = (1 - np.square(np.tanh(ws))) * alpha

            # Update weights
            wNew = (trial1 * wg.T).mean(axis=1) - wg_.mean() * w.squeeze()

            # Decorrelate weights              
            wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])
            wNew = wNew / np.sqrt((wNew ** 2).sum())

            # Calculate limit condition
            lim = np.abs(np.abs((wNew * w).sum()) - 1)
            
            # Update weights
            w = wNew
            
            # Update counter
            i += 1

        W[c, :] = w.T

#%%#Seperated signals   
Seperated_sig = Whitened_sig.T.dot(W.T)
Seperated_sig = (Seperated_sig.T - mean)

#%% Results! for C4
fig, ax = plt.subplots(1, 1, figsize=[18, 5])
ax.plot(trial1[12], lw=5)

ax.set_title('Raw signal')
ax.set_xlim(15, 50)

fig, ax = plt.subplots(1, 1, figsize=[18, 5])

ax.plot(Seperated_sig[12], label='Recovered signals')
ax.set_xlabel('Sample N' )
ax.set_title('Seperated signals' )
ax.set_xlim(15, 50)

plt.show()








