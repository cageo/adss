#/usr/bin/python
# -*- coding: utf-8 -*-
"""
Angular Distance Sample Selection

@author: José Manuel Fernández Jaramillo,
josemanuelfernandezjaramillo-at-yahoo.com

This code is provided under the terms of the:
GNU LESSER GENERAL PUBLIC LICENSE
Version 3, 29 June 2007
"""

import numpy as np

def adss(samples, ncandidates, alpha, units='rads', sort=True):
    """
        samples: sample list as np.array
        ncandidates: number candidates for the training
        alpha: samples inside this angle are rejected 0-> accept all samples pi-> reject all
               samples
        return: test samples, validation samples, w samples
        Define the tolerance in rads, grads or as cosine
    """
    if units == 'rads':
        cosalpha = np.cos(alpha)
    elif units == 'grads':
        cosalpha = np.cos(alpha*np.pi/180)
    elif units == 'cos':
        cosalpha = alpha


    ncandidates = np.min([ncandidates, samples.shape[0]])
    #split the samples in calibration and validation series
    candidates = samples[0:ncandidates]
    wsamples = samples[-(samples.shape[0]-ncandidates):]

    candidatem = np.array(candidates).mean(axis=0)
    candidatesp = candidates-candidatem

    if sort is True:
        norms = np.zeros([candidatesp.shape[0]])
        for i, item in enumerate(norms):
            norms[i] = np.linalg.norm(candidatesp[i, :])
        candidatesp = candidatesp[norms.argsort()[::-1], :]

    testp = []
    validationp = []

    testp.append(candidatesp[0, :])

    for icandidate in range(1, candidatesp.shape[0]):

        candidate = candidatesp[icandidate, :]
        flagacceptintest = 1
        #this for loop is worth to convert in parallel code
        for testsample in testp:
            cosv = np.dot(candidate, testsample)/  \
                   (np.linalg.norm(candidate)*np.linalg.norm(testsample))
            if cosv > cosalpha:
                flagacceptintest = 0
                break

        if flagacceptintest == 1:
            testp.append(candidate)
        else:
            validationp.append(candidate)

    #add the mean and return the sample sets
    test = np.array(testp)
    validation = np.array(validationp)

    if testp:
        test = test+candidatem

    if validationp:
        validation = validation+candidatem

    return test, validation, wsamples





if __name__ == "__main__":
    import matplotlib.pyplot as plt

    TOTALSAMPLES = 1000
    NDIM = 3

    INDEPVAR = 2*np.random.rand(TOTALSAMPLES, NDIM-1)
    DEPVAR = INDEPVAR.dot(np.ones([NDIM-1, 1]))
    SAMPLES = np.c_[INDEPVAR, DEPVAR]

    #samples=np.array([[1,2,3],[4,5,6],[7,-8,9],[-3,4,5],[7,-8,9.01]])

    TSAMPLES, VSAMPLES, WSAMPLES = adss(SAMPLES, np.int(.5*TOTALSAMPLES), 0, sort=True, \
                                       units='grads')

    plt.plot(SAMPLES[:, 0], SAMPLES[:, 1], '.')
    plt.plot(TSAMPLES[:, 0], TSAMPLES[:, 1], 'x')
