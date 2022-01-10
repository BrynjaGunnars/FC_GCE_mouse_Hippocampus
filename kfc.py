#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 21:13:43 2020
@author: brynjagunnarsdottir

Note - Both BootstrapClusterG and makeConsensusMat have a long run time
"""

import numpy as np
from numpy import loadtxt
from sklearn.cluster import SpectralClustering, KMeans
import seaborn as sns
import random
import pandas as pd
from pathlib import Path

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
sns.set()
np.seterr(invalid='ignore')

#Bootstrap clustering function

def BootstrapClusterG(voxelN, k, n, resampleSize, startMouse, endMouse, side, gradientNo):
    
    #storaGeymsla where 1st dimension is voxel number, 2nd dimension is for different k-clustering results(k=2,3,...), 3rd dimension
    #is for different bootstrapped groups
    #litlaGeymsla for keeping bootstrapped groups before k-clustering
    storaGeymsla = np.zeros((voxelN, k-1, n))
    litlaGeymsla = np.zeros((voxelN,n))

    #Make a bootstrapped group average eigenvector n-times
    i=0
    while i<n:
        sub = 0
        eigenvSafn = np.zeros(voxelN)
        #Pick random mouse from  group resampleSize times and save average gradient/eigenvector
        while sub < resampleSize:
            mouseNo = str(random.randint(startMouse,endMouse))
            eigenvName = 'mouse_'+ mouseNo + '_' + side + '_eigenvector_' + gradientNo + '.txt'    
            mouseName = str(Path.cwd()/ eigenvName )
            #mouseName =  '/path/to/'+ '/' mouseNo + '/allen_hippocampus_nosub_'+ side + '_1mm_erode_RS'+ gradientNo +'.eigenvector.txt'
            eigenv = np.loadtxt(fname = mouseName)
            eigenvSafn = eigenvSafn + eigenv
            sub+=1
        eigenvSafn = eigenvSafn/resampleSize
        litlaGeymsla[:,i] = eigenvSafn
        i+=1
        
    i=0
    while i<n:
        cluster=2
        while cluster <= k:
            eigenv = litlaGeymsla[:,i]
            eigenv = eigenv.reshape(-1,1)
            #Initialize k-means
            km = KMeans(init = 'k-means++', n_clusters = cluster )
            km.fit(eigenv)
            clustered = np.array(km.labels_)
            lengd = clustered.size
            j=0
            while j<lengd:
                if clustered[j] == 0:
                    clustered[j]=cluster
                j+=1
            storaGeymsla[:,cluster-2,i]=clustered
            cluster+=1
        i+=1
        
    return storaGeymsla.astype(int)

#Example how to use            
#Left hippocampus, gradient 1, mouse 1-5
#To test remember to set working directory to where eigenvectors text files from test_data are kept

#g1al = BootstrapClusterG(441, 10, 500, 5, 1, 5, 'LEFT', '1')


#np.save('path/to/results/BootstrapMatrix_G1_Left', g1al, allow_pickle=True, fix_imports=True)

#Function to make consensus matrix    
    
def makeConsensusMat(mat): 
    inputMat = np.load(mat)
    matShape = inputMat.shape      
    consensusMat=np.zeros((matShape[0],matShape[0],matShape[1]))
    solution = 2       
    while solution <= (matShape[1]+1):
    
        adjacencyMat = np.zeros((matShape[0], matShape[0]))
        currSolutionMat = np.squeeze(inputMat[:,solution-2,:])
        sub = 0
        while sub < matShape[2]:
            i=0
            while i<matShape[0]:
                j=i
                while j<matShape[0]:
                    if currSolutionMat[i,sub] == currSolutionMat[j,sub]:
                        adjacencyMat[i,j] = adjacencyMat[i,j]+1
                        if i != j:
                            adjacencyMat[j,i] = adjacencyMat[j,i]+1
                    j+=1
                i+=1
            sub+=1
        consensusMat[:,:,solution-2] = adjacencyMat/matShape[2]
        solution+=1
    return consensusMat


#Example how to use


#Consensus matrix all, G1 Left

#m1al = makeConsensusMat('BootstrapMatrix_G1_Left.npy')

#np.save('path/to/results/ConsensusMatrix_G1_Left', m1al, allow_pickle=True, fix_imports=True)


#Function to evaluate different clustering solutions, finds proportion of ambiguous clustering (PAC)

def PAC(mat):
    inputMat= np.load(mat)
    matShape = inputMat.shape
    safnaSaman = np.zeros(matShape[2])
    k=0
    while k < matShape[2]:
        counter = np.count_nonzero(np.logical_and( inputMat[:,:,k] < 0.90, inputMat[:,:,k] > 0.10))
        safnaSaman[k] = counter/(matShape[0]*matShape[1])
        k+=1
    return safnaSaman

#Example how to use

PAC('ConsensusMatrix_G1_Left.npy')
#Result should be: array([0.004, 0.349, 0.287, 0.232, 0.216, 0.219, 0.197, 0.092, 0.102])

#Calculate percent agreement, example 

#no1 = np.load('path/to/ConsensusMatrix_Expl_G1_Left.npy')
#sc1 = SpectralClustering(n_clusters = 2, affinity='precomputed')
#sc1.fit(no1[:,:,0])
#sclustered1 = np.array(sc1.labels_)
#no2 = np.load('path/to/ConsensusMatrix_Val_G1_Left.npy')
#sc2 = SpectralClustering(n_clusters = 2, affinity='precomputed')
#sc2.fit(no1[:,:,0])
#sclustered2 = np.array(sc2.labels_)
#raw_data = {'Expl': sclustered1, 'Val': sclustered2}
#df = pd.DataFrame(raw_data, columns = ['Expl', 'Val'])
#pd.crosstab(df.Expl, df.Val)
