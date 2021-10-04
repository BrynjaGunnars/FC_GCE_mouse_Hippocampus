#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 21:13:43 2020

@author: brynjagunnarsdottir
"""

#set directory
#cd directory


import matplotlib.pyplot as plt
import os
import numpy as np
import nibabel
from numpy import loadtxt
from numpy import corrcoef
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from sklearn.datasets.samples_generator import make_circles
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
import networkx as nx
import seaborn as sns
sns.set()
import nibabel as nib
import sys
import errno
np.seterr(invalid='ignore')
import netneurotools
from netneurotools import stats as nnstats
import scipy
import statsmodels
import statsmodels.api as sm
import random


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
            mouseName =  '/path/to/'+ mouseNo + '/left/or/right/'+ side + '/gradient/number/'+ gradientNo +'.eigenvector.txt'
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
#Left hippocampus, G1 all dataset 

g1al = BootstrapClusterG(441, 10, 500, 50, 1, 50, 'LEFT', '1')

np.save('path/to/results/BootstrapMatrix_All_G1_Left', g1al, allow_pickle=True, fix_imports=True)

#Right hippocampus, G1, all dataset 

g1ar = BootstrapClusterG(436, 10, 500, 50, 1, 50, 'RIGHT', '1')

np.save('path/to/results/BootstrapMatrix_All_G1_Right', g1ar, allow_pickle=True, fix_imports=True)


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

m1al = makeConsensusMat('KFC/BootstrapMatrix_All_G1_Right.npy')

np.save('path/to/results/ConsensusMatrix_All_G1_Left', m1al, allow_pickle=True, fix_imports=True)

#Consensus matrix all, right, G1

m1ar = makeConsensusMat('KFC/BootstrapMatrix_All_G1_Right.npy')

np.save('path/to/results/KFC/ConsensusMatrix_All_G1_Right', m1ar, allow_pickle=True, fix_imports=True)



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



#Heatmap example

n = np.load('path/to/ConsensusMatrix')
plt.imshow(n[:,:,1])
plt.colorbar()
plt.show()
display = sns.clustermap(n[:,:,0])
display.savefig('path/to/results/Heatmap_two_clusters.png')
display = sns.clustermap(n[:,:,1])
display.savefig('path/to/results/Heatmap_three_clusters.png')



#Calculate percent agreement, example 

no1 = np.load('path/to/ConsensusMatrix_Expl_G1_Left.npy')
sc1 = SpectralClustering(n_clusters = 2, affinity='precomputed')
sc1.fit(no1[:,:,0])
sclustered1 = np.array(sc1.labels_)
no2 = np.load('path/to/ConsensusMatrix_Val_G1_Left.npy')
sc2 = SpectralClustering(n_clusters = 2, affinity='precomputed')
sc2.fit(no1[:,:,0])
sclustered2 = np.array(sc2.labels_)
raw_data = {'Expl': sclustered1, 'Val': sclustered2}
df = pd.DataFrame(raw_data, columns = ['Expl', 'Val'])
pd.crosstab(df.Expl, df.Val)


#k-means clustering, example

#Left hippocampus, gradient 1

#Load eigenvector corresponding to gradient 1 to k-means cluster

eigenv1 = np.loadtxt(fname ="path/to/eigenvector/for/gradient1.txt")

#Initialize k-means with number of clusters

km = KMeans(init='k-means++', n_clusters=3)

#Reshape eigenv for k-means to work

eigenv1 = eigenv1.reshape(-1,1)
eigenv1.shape

#K-cluster

km.fit(eigenv1)

#Save labelled vector, change cluster labels to 1,2 and 3 for visualization

clustered1 = np.array(km.labels_)
clustered1.shape
lengd = clustered1.size
lengd
clustered1= clustered1.reshape(lengd,1)
lag = clustered1.shape
i=0
while i < lengd:
    if clustered1[i] == 2:
        clustered1[i]= 3
    elif clustered1[i] == 1:
        clustered1[i] = 2
    elif clustered1[i] == 0:
        clustered1[i] = 1
    i += 1

    

#save clustered vector into image

#Load ROI mask

roiImg = nib.load('path/to/ROI/mask')
roi = roiImg.get_fdata()


#store dimensions of ROI
roidims = roi.shape
roidims
nVoxels = np.prod(roidims)

# Reshape roi into a vector of size nVoxels
roi = np.reshape(roi,(nVoxels))	

# Find the indices inside roi
roiIndices = np.where(roi>0)

outfile = 'path/to/results'+'name'+'nii.gz'

yDat = np.zeros(shape=roidims+(1,))
yDat = np.reshape(yDat, (np.prod(roidims), 1))
yDat[roiIndices,:] = clustered1
yDat = np.reshape(yDat,roidims)
yImg = nib.Nifti1Image(yDat,roiImg.get_affine(),roiImg.get_header())


nib.save(yImg,outfile)


#Spectral clustering, example


m1al = np.load('path/to/ConsensusMatrix_All_G3_Right.npy')
sc = SpectralClustering(n_clusters = 2, affinity='precomputed')

sc.fit(m1al)
sclustered = np.array(sc.labels_)
sclustered.shape
sclustered
j=0
while j<sclustered.shape[0]:
    if sclustered[j] == 0:
        sclustered[j]=2
    j+=1
    
#Example how to save mask of a cluster, fslmaths is afterwards used to label the cluster correctly and
#to replace clusters with the hippocampus subfields in the Allen Common Coordinate Framework

#maskmaker cluster 2 - label 2

lengd=sclustered.size
sclustered=sclustered.reshape(lengd,1)
clustermask = np.zeros(shape = sclustered.shape)
clustermask.shape
lengd
i=0
while i < lengd:
    if sclustered[i] == 2:
        clustermask[i]= 1
    else :
        clustermask[i] = 0
    i += 1


#save clusteredmask into image

#Load ROI mask

roiImg = nib.load('path/to/ROI/mask')
roi = roiImg.get_fdata()


#store dimensions of ROI
roidims = roi.shape
roidims
nVoxels = np.prod(roidims)

# Reshape roi into a vector of size nVoxels
roi = np.reshape(roi,(nVoxels))	

# Find the indices inside roi
roiIndices = np.where(roi>0)

outfile = 'path/to/results' + 'name' + '.nii.gz'

yDat = np.zeros(shape=roidims+(1,))
yDat = np.reshape(yDat, (np.prod(roidims), 1))
yDat[roiIndices,:] = clustermask
yDat = np.reshape(yDat,roidims)
yImg = nib.Nifti1Image(yDat,roiImg.get_affine(),roiImg.get_header())

nib.save(yImg,outfile)

#Function to make Correlation matrix

#Load atlas w hippocampus
#Side 1 = left, side 2= right hippocampus, G=0 for anatomical atlas, G=1/2/3 for gradient 1 and 2 and 3
#mouseno is the mouse number in the form of a string e.g. '001'

def CorrMatrixMaker(mouseno, G, side):
    
    if G == 0:
        hippoAtlas = nib.load('allen_parcellation_zerbi165_subMASKED.nii.gz')
        hippoAtlas = hippoAtlas.get_fdata()
        hippoAtlas_filename = 'allen_parcellation_zerbi165_subMASKED.nii.gz'
    elif G == 1:
        if side == 1:
            hippoAtlas = nib.load('allen_parcellation_zerbi165_subMASKED_G1_Left.nii.gz')
            hippoAtlas = hippoAtlas.get_fdata()
            hippoAtlas_filename = 'allen_parcellation_zerbi165_subMASKED_G1_Left.nii.gz'
        elif side ==2:
            hippoAtlas = nib.load('allen_parcellation_zerbi165_subMASKED_G1_Right.nii.gz')
            hippoAtlas = hippoAtlas.get_fdata()
            hippoAtlas_filename = 'allen_parcellation_zerbi165_subMASKED_G1_Right.nii.gz'
           
    elif G ==2:
        if side == 1:
            hippoAtlas = nib.load('allen_parcellation_zerbi165_subMASKED_G2_Left.nii.gz')
            hippoAtlas = hippoAtlas.get_fdata()
            hippoAtlas_filename = 'allen_parcellation_zerbi165_subMASKED_G2_Left.nii.gz'
        elif side ==2:
            hippoAtlas = nib.load('allen_parcellation_zerbi165_subMASKED_G2_Right.nii.gz')
            hippoAtlas = hippoAtlas.get_fdata()
            hippoAtlas_filename = 'allen_parcellation_zerbi165_subMASKED_G2_Right.nii.gz'
    elif G ==3:
        if side == 1:
            hippoAtlas = nib.load('allen_parcellation_zerbi165_subMASKED_G3_Left.nii.gz')
            hippoAtlas = hippoAtlas.get_fdata()
            hippoAtlas_filename = 'allen_parcellation_zerbi165_subMASKED_G3_Left.nii.gz'
        elif side ==2:
            hippoAtlas = nib.load('allen_parcellation_zerbi165_subMASKED_G3_Right.nii.gz')
            hippoAtlas = hippoAtlas.get_fdata()
            hippoAtlas_filename = 'allen_parcellation_zerbi165_subMASKED_G3_Right.nii.gz'        

    #Load mouse 1 fmri run
        
    mouseRun_filename = mouseno + '_FIXed_QBI.nii.gz'
    mouseRun = nib.load(mouseRun_filename)
    mouseRun = mouseRun.get_fdata()

    #Extract signals from each parcel

    from nilearn.input_data import NiftiLabelsMasker

    #make mask object to extract signals from each parcellation

    masker = NiftiLabelsMasker(labels_img=hippoAtlas_filename, standardize=True, memory='nilearn_cache', verbose=5)

    # Here we go from nifti files to the signal time series in a numpy
    # array. Note how we give confounds to be regressed out during signal
    # extraction

    atlasExtract = masker.fit_transform(mouseRun_filename)

    #Make correlation matrix

    from nilearn.connectome import ConnectivityMeasure
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([atlasExtract])[0]
    if G == 0:
        saveName = 'KFC/corr_matrix_'+mouseno+'_w_hippocampus'
    elif G==1:
        if side == 1:
            saveName = 'KFC/corr_matrix_'+mouseno+'_G1_clustered_Left'
        elif side ==2:
            saveName = 'KFC/corr_matrix_'+mouseno+'_G1_clustered_Right'
    elif G==2:
        if side == 1:
            saveName = 'KFC/corr_matrix_'+mouseno+'_G2_clustered_Left'
        elif side ==2:
            saveName = 'KFC/corr_matrix_'+mouseno+'_G2_clustered_Right'
    elif G==3:
        if side == 1:
            saveName = 'KFC/corr_matrix_'+mouseno+'_G3_clustered_Left'
        elif side ==2:
            saveName = 'KFC/corr_matrix_'+mouseno+'_G3_clustered_Right'
    np.save(saveName, correlation_matrix, allow_pickle=True, fix_imports=True)
    return correlation_matrix
 

#Make correlation matrix, not the smart way/not using function. Example for anatomical 
#correlations   

#Mouse 002

#Load atlas w hippocampus


hippoAtlas = nib.load('allen_parcellation_zerbi165_subMASKED.nii')
hippoAtlas = hippoAtlas.get_fdata()
hippoAtlas_filename = 'allen_parcellation_zerbi165_subMASKED.nii'

#Load mouse fmri run

mouseRun = nib.load('002_FIXed_QBI.nii.gz')
mouseRun = mouseRun.get_fdata()
mouseRun_filename = '002_FIXed_QBI.nii.gz'

#Extract signals from each parcel

from nilearn.input_data import NiftiLabelsMasker

#make mask object to extract signals from each parcellation

masker = NiftiLabelsMasker(labels_img=hippoAtlas_filename, standardize=True, memory='nilearn_cache', verbose=5)

# Here we go from nifti files to the signal time series in a numpy
# array. Note how we give confounds to be regressed out during signal
# extraction

atlasExtract = masker.fit_transform(mouseRun_filename)

#Make correlation matrix

from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([atlasExtract])[0]
np.save('KFC/corr_matrix_002_w_hippocampus', correlation_matrix, allow_pickle=True, fix_imports=True)

# Plot the correlation matrix
import numpy as np
from nilearn import plotting
# Make a large figure
# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix, 0)
# The labels we have start with the background (0), hence we skip the
# first label
# matrices are ordered for block-like representation
#try to make labels
import pandas as pd

df = pd.read_excel('ROI_List_Allen_213_to_165.xlsx')
df.as_matrix()
allenAtlas = df.to_numpy()


display = plotting.plot_matrix(correlation_matrix, figure=(50, 40), labels=allenAtlas[:,4],vmax=0.8, vmin=-0.8)
display.figure.savefig('KFC/corr_img_002_w_hippo.png')




#Example how to find significant correlations for gradient 1 in left hemisphere

#Matrix for significant restults

#Load correlation matrices for gradient 1 in left hemisphere

#001

allen001_G1_L = np.load('KFC/corr_matrix_001_G1_clustered_Left.npy')

allen001_G1_L_48 = allen001_G1_L[47,:]

allen001_G1_L_49 = allen001_G1_L[48,:]


#002

allen002_G1_L = np.load('KFC/corr_matrix_002_G1_clustered_Left.npy')

allen002_G1_L_48 = allen002_G1_L[47,:]

allen002_G1_L_49 = allen002_G1_L[48,:]



#003

allen003_G1_L = np.load('KFC/corr_matrix_003_G1_clustered_Left.npy')

allen003_G1_L_48 = allen003_G1_L[47,:]

allen003_G1_L_49 = allen003_G1_L[48,:]



#004

allen004_G1_L = np.load('KFC/corr_matrix_004_G1_clustered_Left.npy')

allen004_G1_L_48 = allen004_G1_L[47,:]

allen004_G1_L_49 = allen004_G1_L[48,:]



#005

allen005_G1_L = np.load('KFC/corr_matrix_005_G1_clustered_Left.npy')

allen005_G1_L_48 = allen005_G1_L[47,:]

allen005_G1_L_49 = allen005_G1_L[48,:]



#006

allen006_G1_L = np.load('KFC/corr_matrix_006_G1_clustered_Left.npy')

allen006_G1_L_48 = allen006_G1_L[47,:]

allen006_G1_L_49 = allen006_G1_L[48,:]



#007

allen007_G1_L = np.load('KFC/corr_matrix_007_G1_clustered_Left.npy')

allen007_G1_L_48 = allen007_G1_L[47,:]

allen007_G1_L_49 = allen007_G1_L[48,:]



#008

allen008_G1_L = np.load('KFC/corr_matrix_008_G1_clustered_Left.npy')

allen008_G1_L_48 = allen008_G1_L[47,:]

allen008_G1_L_49 = allen008_G1_L[48,:]



#009

allen009_G1_L = np.load('KFC/corr_matrix_009_G1_clustered_Left.npy')

allen009_G1_L_48 = allen009_G1_L[47,:]

allen009_G1_L_49 = allen009_G1_L[48,:]


#010

allen010_G1_L = np.load('KFC/corr_matrix_010_G1_clustered_Left.npy')

allen010_G1_L_48 = allen010_G1_L[47,:]

allen010_G1_L_49 = allen010_G1_L[48,:]


#011
allen011_G1_L = np.load('KFC/corr_matrix_011_G1_clustered_Left.npy')

allen011_G1_L_48 = allen011_G1_L[47,:]

allen011_G1_L_49 = allen011_G1_L[48,:]

#012
allen012_G1_L = np.load('KFC/corr_matrix_012_G1_clustered_Left.npy')

allen012_G1_L_48 = allen012_G1_L[47,:]

allen012_G1_L_49 = allen012_G1_L[48,:]



#013
allen013_G1_L = np.load('KFC/corr_matrix_013_G1_clustered_Left.npy')

allen013_G1_L_48 = allen013_G1_L[47,:]

allen013_G1_L_49 = allen013_G1_L[48,:]



#014
allen014_G1_L = np.load('KFC/corr_matrix_014_G1_clustered_Left.npy')

allen014_G1_L_48 = allen014_G1_L[47,:]

allen014_G1_L_49 = allen014_G1_L[48,:]



#015
allen015_G1_L = np.load('KFC/corr_matrix_015_G1_clustered_Left.npy')

allen015_G1_L_48 = allen015_G1_L[47,:]

allen015_G1_L_49 = allen015_G1_L[48,:]



#016
allen016_G1_L = np.load('KFC/corr_matrix_016_G1_clustered_Left.npy')

allen016_G1_L_48 = allen016_G1_L[47,:]

allen016_G1_L_49 = allen016_G1_L[48,:]



#017
allen017_G1_L = np.load('KFC/corr_matrix_017_G1_clustered_Left.npy')

allen017_G1_L_48 = allen017_G1_L[47,:]

allen017_G1_L_49 = allen017_G1_L[48,:]



#018
allen018_G1_L = np.load('KFC/corr_matrix_018_G1_clustered_Left.npy')

allen018_G1_L_48 = allen018_G1_L[47,:]

allen018_G1_L_49 = allen018_G1_L[48,:]



#019
allen019_G1_L = np.load('KFC/corr_matrix_019_G1_clustered_Left.npy')

allen019_G1_L_48 = allen019_G1_L[47,:]

allen019_G1_L_49 = allen019_G1_L[48,:]



#020
allen020_G1_L = np.load('KFC/corr_matrix_020_G1_clustered_Left.npy')

allen020_G1_L_48 = allen020_G1_L[47,:]

allen020_G1_L_49 = allen020_G1_L[48,:]



#021
allen021_G1_L = np.load('KFC/corr_matrix_021_G1_clustered_Left.npy')

allen021_G1_L_48 = allen021_G1_L[47,:]

allen021_G1_L_49 = allen021_G1_L[48,:]



#022
allen022_G1_L = np.load('KFC/corr_matrix_022_G1_clustered_Left.npy')

allen022_G1_L_48 = allen022_G1_L[47,:]

allen022_G1_L_49 = allen022_G1_L[48,:]



#023
allen023_G1_L = np.load('KFC/corr_matrix_023_G1_clustered_Left.npy')

allen023_G1_L_48 = allen023_G1_L[47,:]

allen023_G1_L_49 = allen023_G1_L[48,:]


#024
allen024_G1_L = np.load('KFC/corr_matrix_024_G1_clustered_Left.npy')

allen024_G1_L_48 = allen024_G1_L[47,:]

allen024_G1_L_49 = allen024_G1_L[48,:]



#025
allen025_G1_L = np.load('KFC/corr_matrix_025_G1_clustered_Left.npy')

allen025_G1_L_48 = allen025_G1_L[47,:]

allen025_G1_L_49 = allen025_G1_L[48,:]



#026
allen026_G1_L = np.load('KFC/corr_matrix_026_G1_clustered_Left.npy')

allen026_G1_L_48 = allen026_G1_L[47,:]

allen026_G1_L_49 = allen026_G1_L[48,:]



#027
allen027_G1_L = np.load('KFC/corr_matrix_027_G1_clustered_Left.npy')

allen027_G1_L_48 = allen027_G1_L[47,:]

allen027_G1_L_49 = allen027_G1_L[48,:]



#028
allen028_G1_L = np.load('KFC/corr_matrix_028_G1_clustered_Left.npy')

allen028_G1_L_48 = allen028_G1_L[47,:]

allen028_G1_L_49 = allen028_G1_L[48,:]



#029
allen029_G1_L = np.load('KFC/corr_matrix_029_G1_clustered_Left.npy')

allen029_G1_L_48 = allen029_G1_L[47,:]

allen029_G1_L_49 = allen029_G1_L[48,:]


#030
allen030_G1_L = np.load('KFC/corr_matrix_030_G1_clustered_Left.npy')

allen030_G1_L_48 = allen030_G1_L[47,:]

allen030_G1_L_49 = allen030_G1_L[48,:]


#031
allen031_G1_L = np.load('KFC/corr_matrix_031_G1_clustered_Left.npy')

allen031_G1_L_48 = allen031_G1_L[47,:]

allen031_G1_L_49 = allen031_G1_L[48,:]


#032
allen032_G1_L = np.load('KFC/corr_matrix_032_G1_clustered_Left.npy')

allen032_G1_L_48 = allen032_G1_L[47,:]

allen032_G1_L_49 = allen032_G1_L[48,:]



#033
allen033_G1_L = np.load('KFC/corr_matrix_033_G1_clustered_Left.npy')

allen033_G1_L_48 = allen033_G1_L[47,:]

allen033_G1_L_49 = allen033_G1_L[48,:]



#034
allen034_G1_L = np.load('KFC/corr_matrix_034_G1_clustered_Left.npy')

allen034_G1_L_48 = allen034_G1_L[47,:]

allen034_G1_L_49 = allen034_G1_L[48,:]



#035
allen035_G1_L = np.load('KFC/corr_matrix_035_G1_clustered_Left.npy')

allen035_G1_L_48 = allen035_G1_L[47,:]

allen035_G1_L_49 = allen035_G1_L[48,:]


#036
allen036_G1_L = np.load('KFC/corr_matrix_036_G1_clustered_Left.npy')

allen036_G1_L_48 = allen036_G1_L[47,:]

allen036_G1_L_49 = allen036_G1_L[48,:]



#037
allen037_G1_L = np.load('KFC/corr_matrix_037_G1_clustered_Left.npy')

allen037_G1_L_48 = allen037_G1_L[47,:]

allen037_G1_L_49 = allen037_G1_L[48,:]



#038
allen038_G1_L = np.load('KFC/corr_matrix_038_G1_clustered_Left.npy')

allen038_G1_L_48 = allen038_G1_L[47,:]

allen038_G1_L_49 = allen038_G1_L[48,:]



#039
allen039_G1_L = np.load('KFC/corr_matrix_039_G1_clustered_Left.npy')

allen039_G1_L_48 = allen039_G1_L[47,:]

allen039_G1_L_49 = allen039_G1_L[48,:]



#040
allen040_G1_L = np.load('KFC/corr_matrix_040_G1_clustered_Left.npy')

allen040_G1_L_48 = allen040_G1_L[47,:]

allen040_G1_L_49 = allen040_G1_L[48,:]



#041
allen041_G1_L = np.load('KFC/corr_matrix_041_G1_clustered_Left.npy')

allen041_G1_L_48 = allen041_G1_L[47,:]

allen041_G1_L_49 = allen041_G1_L[48,:]



#042
allen042_G1_L = np.load('KFC/corr_matrix_042_G1_clustered_Left.npy')

allen042_G1_L_48 = allen042_G1_L[47,:]

allen042_G1_L_49 = allen042_G1_L[48,:]



#043
allen043_G1_L = np.load('KFC/corr_matrix_043_G1_clustered_Left.npy')

allen043_G1_L_48 = allen043_G1_L[47,:]

allen043_G1_L_49 = allen043_G1_L[48,:]



#044
allen044_G1_L = np.load('KFC/corr_matrix_044_G1_clustered_Left.npy')

allen044_G1_L_48 = allen044_G1_L[47,:]

allen044_G1_L_49 = allen044_G1_L[48,:]



#045
allen045_G1_L = np.load('KFC/corr_matrix_045_G1_clustered_Left.npy')

allen045_G1_L_48 = allen045_G1_L[47,:]

allen045_G1_L_49 = allen045_G1_L[48,:]



#046
allen046_G1_L = np.load('KFC/corr_matrix_046_G1_clustered_Left.npy')

allen046_G1_L_48 = allen046_G1_L[47,:]

allen046_G1_L_49 = allen046_G1_L[48,:]



#047
allen047_G1_L = np.load('KFC/corr_matrix_047_G1_clustered_Left.npy')

allen047_G1_L_48 = allen047_G1_L[47,:]

allen047_G1_L_49 = allen047_G1_L[48,:]



#048
allen048_G1_L = np.load('KFC/corr_matrix_048_G1_clustered_Left.npy')

allen048_G1_L_48 = allen048_G1_L[47,:]

allen048_G1_L_49 = allen048_G1_L[48,:]



#049
allen049_G1_L = np.load('KFC/corr_matrix_049_G1_clustered_Left.npy')

allen049_G1_L_48 = allen049_G1_L[47,:]

allen049_G1_L_49 = allen049_G1_L[48,:]



#050
allen050_G1_L = np.load('KFC/corr_matrix_050_G1_clustered_Left.npy')

allen050_G1_L_48 = allen050_G1_L[47,:]

allen050_G1_L_49 = allen050_G1_L[48,:]




#Make array with significant results, if significant save mean correlation across mice

#Cluster label 48, left hippocampus G1

#Matrix for significant restults

staerd = allen001_G1_L_48.size
staerd
significant_G1_L_48 = np.zeros(staerd)

#Matrix with correlation between ROI and other ROI across mice

stok = np.zeros(50) 

#Matrix for Benjamini/Hochberg corrected significance

leidrett = np.zeros(staerd)

m = 0
while m < staerd:
    stok[0]=allen001_G1_L_48[m]
    stok[1]=allen002_G1_L_48[m]
    stok[2]=allen003_G1_L_48[m]
    stok[3]=allen004_G1_L_48[m]
    stok[4]=allen005_G1_L_48[m]
    stok[5]=allen006_G1_L_48[m]
    stok[6]=allen007_G1_L_48[m]
    stok[7]=allen008_G1_L_48[m]
    stok[8]=allen009_G1_L_48[m]
    stok[9]=allen010_G1_L_48[m]
    stok[10]=allen011_G1_L_48[m]
    stok[11]=allen012_G1_L_48[m]
    stok[12]=allen013_G1_L_48[m]
    stok[13]=allen014_G1_L_48[m]
    stok[14]=allen015_G1_L_48[m]
    stok[15]=allen016_G1_L_48[m]
    stok[16]=allen017_G1_L_48[m]
    stok[17]=allen018_G1_L_48[m]
    stok[18]=allen019_G1_L_48[m]
    stok[19]=allen020_G1_L_48[m]
    stok[20]=allen021_G1_L_48[m]
    stok[21]=allen022_G1_L_48[m]
    stok[22]=allen023_G1_L_48[m]
    stok[23]=allen024_G1_L_48[m]
    stok[24]=allen025_G1_L_48[m]
    stok[25]=allen026_G1_L_48[m]
    stok[26]=allen027_G1_L_48[m]
    stok[27]=allen028_G1_L_48[m]
    stok[28]=allen029_G1_L_48[m]
    stok[29]=allen030_G1_L_48[m]
    stok[30]=allen031_G1_L_48[m]
    stok[31]=allen032_G1_L_48[m]
    stok[32]=allen033_G1_L_48[m]
    stok[33]=allen034_G1_L_48[m]
    stok[34]=allen035_G1_L_48[m]
    stok[35]=allen036_G1_L_48[m]
    stok[36]=allen037_G1_L_48[m]
    stok[37]=allen038_G1_L_48[m]
    stok[38]=allen039_G1_L_48[m]
    stok[39]=allen040_G1_L_48[m]
    stok[40]=allen041_G1_L_48[m]
    stok[41]=allen042_G1_L_48[m]
    stok[42]=allen043_G1_L_48[m]
    stok[43]=allen044_G1_L_48[m]
    stok[44]=allen045_G1_L_48[m]
    stok[45]=allen046_G1_L_48[m]
    stok[46]=allen047_G1_L_48[m]
    stok[47]=allen048_G1_L_48[m]
    stok[48]=allen049_G1_L_48[m]
    stok[49]=allen050_G1_L_48[m] 
    
    #Perform non-parametric 1 sample permutation test to ascertain correlation is significant
    
    out = nnstats.permtest_1samp(stok, 0.0)
    
    leidrett[m] = out[1]
    
    m+=1
    
leidrett = statsmodels.stats.multitest.multipletests(leidrett, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)

#Save boolean matrix of correction, true if result is significant, false if not

leidrett = leidrett[0]
leidrett

#Save significant results after correction

m = 0
while m < staerd:
    stok[0]=allen001_G1_L_48[m]
    stok[1]=allen002_G1_L_48[m]
    stok[2]=allen003_G1_L_48[m]
    stok[3]=allen004_G1_L_48[m]
    stok[4]=allen005_G1_L_48[m]
    stok[5]=allen006_G1_L_48[m]
    stok[6]=allen007_G1_L_48[m]
    stok[7]=allen008_G1_L_48[m]
    stok[8]=allen009_G1_L_48[m]
    stok[9]=allen010_G1_L_48[m]
    stok[10]=allen011_G1_L_48[m]
    stok[11]=allen012_G1_L_48[m]
    stok[12]=allen013_G1_L_48[m]
    stok[13]=allen014_G1_L_48[m]
    stok[14]=allen015_G1_L_48[m]
    stok[15]=allen016_G1_L_48[m]
    stok[16]=allen017_G1_L_48[m]
    stok[17]=allen018_G1_L_48[m]
    stok[18]=allen019_G1_L_48[m]
    stok[19]=allen020_G1_L_48[m]
    stok[20]=allen021_G1_L_48[m]
    stok[21]=allen022_G1_L_48[m]
    stok[22]=allen023_G1_L_48[m]
    stok[23]=allen024_G1_L_48[m]
    stok[24]=allen025_G1_L_48[m]
    stok[25]=allen026_G1_L_48[m]
    stok[26]=allen027_G1_L_48[m]
    stok[27]=allen028_G1_L_48[m]
    stok[28]=allen029_G1_L_48[m]
    stok[29]=allen030_G1_L_48[m]
    stok[30]=allen031_G1_L_48[m]
    stok[31]=allen032_G1_L_48[m]
    stok[32]=allen033_G1_L_48[m]
    stok[33]=allen034_G1_L_48[m]
    stok[34]=allen035_G1_L_48[m]
    stok[35]=allen036_G1_L_48[m]
    stok[36]=allen037_G1_L_48[m]
    stok[37]=allen038_G1_L_48[m]
    stok[38]=allen039_G1_L_48[m]
    stok[39]=allen040_G1_L_48[m]
    stok[40]=allen041_G1_L_48[m]
    stok[41]=allen042_G1_L_48[m]
    stok[42]=allen043_G1_L_48[m]
    stok[43]=allen044_G1_L_48[m]
    stok[44]=allen045_G1_L_48[m]
    stok[45]=allen046_G1_L_48[m]
    stok[46]=allen047_G1_L_48[m]
    stok[47]=allen048_G1_L_48[m]
    stok[48]=allen049_G1_L_48[m]
    stok[49]=allen050_G1_L_48[m]     
    
    #If signifant and positively correlated save mean of correlation across mice,
    #otherwise put zero in the significant result matrix, zero autocorrelation
    
    if leidrett[m]==True:
        if np.mean(stok) == 1.000:
            significant_G1_L_48[m]=0
        else:
            significant_G1_L_48[m]= np.mean(stok)
    else:
        significant_G1_L_48[m]= 0
    m +=1


#Save significant correlation ROI label

m = 0
listi_G1_L_48 = np.zeros(staerd)
while m < staerd:
    if significant_G1_L_48[m] != 0:
        listi_G1_L_48[m] = m+1
    m+=1
        

#Save list of significant ROI
#
#

#Read ROI list

df = pd.read_excel('ROI_List_Allen_213_to_165_G1_Clustered_Left.xlsx')
df.as_matrix()
allenList = df.to_numpy()
allenList
Dims = allenList.shape
staerd = Dims[0]

#Make list to put significant negative correlation ROI into

nafnalisti_neg = []
i=0
while i < staerd:
    if listi_G1_L_48[i] != 0 and significant_G1_L_48[i] < 0:
        nafnalisti_neg.append(allenList[i,4])
    i+=1
    
    
nafnalisti_neg = np.array(nafnalisti_neg)

#Save as excel file

excel = pd.DataFrame(nafnalisti_neg)
excel.to_excel(excel_writer = "KFC/G1_L_48_neg_corr_prufa.xlsx")

#Make list to put significant positive correlation ROI into


nafnalisti_pos = []
i=0
while i < staerd:
    if listi_G1_L_48[i] != 0 and significant_G1_L_48[i] > 0:
        nafnalisti_pos.append(allenList[i,4])
    i+=1
    
    
nafnaListi_pos = np.array(nafnalisti_pos)
nafnaListi_pos  

#Save as excel file

excel = pd.DataFrame(nafnalisti_pos)
excel.to_excel(excel_writer = "KFC/G1_L_48_pos_corr_prufa.xlsx")
 



#Make image with significant correlations

allenAtlasImg = nib.load('allen_parcellation_zerbi165_subMASKED_G1_Left.nii.gz')
allenAtlas = allenAtlasImg.get_fdata()

#store dimensions of ROI
allenAtlasDim = allenAtlas.shape
allenAtlasDim
nVoxels = np.prod(allenAtlasDim)
nVoxels

# Reshape roi into a vector of size nVoxels
allenAtlas = np.reshape(allenAtlas,(nVoxels))	


# Find the indices inside roi
allenIndices = np.where(allenAtlas>0)

allenTeljari = allenIndices[0]
lengd = np.size(allenIndices)

significantIndices = np.zeros(lengd)

#Iterate through all areas in ROI where label >0, if label is equal to significantly correlated ROI change label into average correlation
#Note:  allenAtlas[allenTeljari[m]] == listiCA2[i] != 0.0 because allenAtlas value with the indices in allenTeljari is by definition >0



m=0
while m < lengd:
    i=0
    while i < staerd:
        if allenAtlas[allenTeljari[m]] == listi_G1_L_48[i]:
            heiltala = int(listi_G1_L_48[i]-1)
            significantIndices[m] = significant_G1_L_48[heiltala]
            break
        else:
            significantIndices[m] = 0
        i+=1
    m+=1

significantIndices = significantIndices.reshape(lengd,1)      
    
    
outfile = 'KFC' + "/" + 'allen_atlas_corr_G1_48_L' + ".nii.gz"

yDat = np.zeros(shape=allenAtlasDim + (1,))
yDat = np.reshape(yDat, (np.prod(allenAtlasDim), 1))
yDat[allenIndices,:] = significantIndices
yDat = np.reshape(yDat,allenAtlasDim)
yImg = nib.Nifti1Image(yDat,allenAtlasImg.get_affine(),allenAtlasImg.get_header())


nib.save(yImg,outfile)


#Make array with significant results, if significant save mean correlation across mice

#Cluster label 49 G1

#Matrix for significant restults

staerd = allen001_G1_L_49.size
significant_G1_49 = np.zeros(staerd)

#Matrix with correlation between ROI and other ROI across mice

stok = np.zeros(50)

#Matrix for Benjamini/Hochberg corrected significance

leidrett = np.zeros(staerd) 

 
m = 0
while m < staerd:
    stok[0]=allen001_G1_L_49[m]
    stok[1]=allen002_G1_L_49[m]
    stok[2]=allen003_G1_L_49[m]
    stok[3]=allen004_G1_L_49[m]
    stok[4]=allen005_G1_L_49[m]
    stok[5]=allen006_G1_L_49[m]
    stok[6]=allen007_G1_L_49[m]
    stok[7]=allen008_G1_L_49[m]
    stok[8]=allen009_G1_L_49[m]
    stok[9]=allen010_G1_L_49[m]
    stok[10]=allen011_G1_L_49[m]
    stok[11]=allen012_G1_L_49[m]
    stok[12]=allen013_G1_L_49[m]
    stok[13]=allen014_G1_L_49[m]
    stok[14]=allen015_G1_L_49[m]
    stok[15]=allen016_G1_L_49[m]
    stok[16]=allen017_G1_L_49[m]
    stok[17]=allen018_G1_L_49[m]
    stok[18]=allen019_G1_L_49[m]
    stok[19]=allen020_G1_L_49[m]
    stok[20]=allen021_G1_L_49[m]
    stok[21]=allen022_G1_L_49[m]
    stok[22]=allen023_G1_L_49[m]
    stok[23]=allen024_G1_L_49[m]
    stok[24]=allen025_G1_L_49[m]
    stok[25]=allen026_G1_L_49[m]
    stok[26]=allen027_G1_L_49[m]
    stok[27]=allen028_G1_L_49[m]
    stok[28]=allen029_G1_L_49[m]
    stok[29]=allen030_G1_L_49[m]
    stok[30]=allen031_G1_L_49[m]
    stok[31]=allen032_G1_L_49[m]
    stok[32]=allen033_G1_L_49[m]
    stok[33]=allen034_G1_L_49[m]
    stok[34]=allen035_G1_L_49[m]
    stok[35]=allen036_G1_L_49[m]
    stok[36]=allen037_G1_L_49[m]
    stok[37]=allen038_G1_L_49[m]
    stok[38]=allen039_G1_L_49[m]
    stok[39]=allen040_G1_L_49[m]
    stok[40]=allen041_G1_L_49[m]
    stok[41]=allen042_G1_L_49[m]
    stok[42]=allen043_G1_L_49[m]
    stok[43]=allen044_G1_L_49[m]
    stok[44]=allen045_G1_L_49[m]
    stok[45]=allen046_G1_L_49[m]
    stok[46]=allen047_G1_L_49[m]
    stok[47]=allen048_G1_L_49[m]
    stok[48]=allen049_G1_L_49[m]
    stok[49]=allen050_G1_L_49[m] 
    
    #Perform non-parametric 1 sample permutation test to ascertain correlation is significant
    
    out = nnstats.permtest_1samp(stok, 0.0)
    
    leidrett[m] = out[1]
    
    m+=1
    
leidrett = statsmodels.stats.multitest.multipletests(leidrett, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)

#Save boolean matrix of correction, true if result is significant, false if not

leidrett = leidrett[0]

#Save significant results after correction

m = 0
while m < staerd:
    stok[0]=allen001_G1_L_49[m]
    stok[1]=allen002_G1_L_49[m]
    stok[2]=allen003_G1_L_49[m]
    stok[3]=allen004_G1_L_49[m]
    stok[4]=allen005_G1_L_49[m]
    stok[5]=allen006_G1_L_49[m]
    stok[6]=allen007_G1_L_49[m]
    stok[7]=allen008_G1_L_49[m]
    stok[8]=allen009_G1_L_49[m]
    stok[9]=allen010_G1_L_49[m]
    stok[10]=allen011_G1_L_49[m]
    stok[11]=allen012_G1_L_49[m]
    stok[12]=allen013_G1_L_49[m]
    stok[13]=allen014_G1_L_49[m]
    stok[14]=allen015_G1_L_49[m]
    stok[15]=allen016_G1_L_49[m]
    stok[16]=allen017_G1_L_49[m]
    stok[17]=allen018_G1_L_49[m]
    stok[18]=allen019_G1_L_49[m]
    stok[19]=allen020_G1_L_49[m]
    stok[20]=allen021_G1_L_49[m]
    stok[21]=allen022_G1_L_49[m]
    stok[22]=allen023_G1_L_49[m]
    stok[23]=allen024_G1_L_49[m]
    stok[24]=allen025_G1_L_49[m]
    stok[25]=allen026_G1_L_49[m]
    stok[26]=allen027_G1_L_49[m]
    stok[27]=allen028_G1_L_49[m]
    stok[28]=allen029_G1_L_49[m]
    stok[29]=allen030_G1_L_49[m]
    stok[30]=allen031_G1_L_49[m]
    stok[31]=allen032_G1_L_49[m]
    stok[32]=allen033_G1_L_49[m]
    stok[33]=allen034_G1_L_49[m]
    stok[34]=allen035_G1_L_49[m]
    stok[35]=allen036_G1_L_49[m]
    stok[36]=allen037_G1_L_49[m]
    stok[37]=allen038_G1_L_49[m]
    stok[38]=allen039_G1_L_49[m]
    stok[39]=allen040_G1_L_49[m]
    stok[40]=allen041_G1_L_49[m]
    stok[41]=allen042_G1_L_49[m]
    stok[42]=allen043_G1_L_49[m]
    stok[43]=allen044_G1_L_49[m]
    stok[44]=allen045_G1_L_49[m]
    stok[45]=allen046_G1_L_49[m]
    stok[46]=allen047_G1_L_49[m]
    stok[47]=allen048_G1_L_49[m]
    stok[48]=allen049_G1_L_49[m]
    stok[49]=allen050_G1_L_49[m] 
       
    #If signifant and positively correlated save mean of correlation across mice,
    #otherwise put zero in the significant result matrix, zero autocorrelation
    
    if leidrett[m]==True:
        if np.mean(stok) == 1.000:
            significant_G1_49[m]=0
        else:
            significant_G1_49[m]= np.mean(stok)
    else:
        significant_G1_49[m]= 0
    m +=1
significant_G1_49

#Save significant correlation ROI label

m = 0
listi_G1_49 = np.zeros(staerd)
while m < staerd:
    if significant_G1_49[m] != 0:
        listi_G1_49[m] = m+1
    m+=1
        
listi_G1_49

#Save list of significant ROI
#
#

#Read ROI list

df = pd.read_excel('ROI_List_Allen_213_to_165_G1_Clustered_Left.xlsx')
df.as_matrix()
allenList = df.to_numpy()
Dims = allenList.shape
staerd = Dims[0]

#Make list to put significant negative correlation ROI into

nafnalisti_neg = []
i=0
while i < staerd:
    if listi_G1_49[i] != 0 and significant_G1_49[i] < 0:
        nafnalisti_neg.append(allenList[i,4])
    i+=1
    
    
nafnalisti_neg = np.array(nafnalisti_neg)
nafnalisti_neg

#Save as excel file

excel = pd.DataFrame(nafnalisti_neg)
excel.to_excel(excel_writer = "KFC/G1_L_49_neg_corr_prufa.xlsx")

#Make list to put significant positive correlation ROI into


nafnalisti_pos = []
i=0
while i < staerd:
    if listi_G1_49[i] != 0 and significant_G1_49[i] > 0:
        nafnalisti_pos.append(allenList[i,4])
    i+=1
    
    
nafnaListi_pos = np.array(nafnalisti_pos)
nafnaListi_pos  

#Save as excel file

excel = pd.DataFrame(nafnalisti_pos)
excel.to_excel(excel_writer = "KFC/G1_L_49_pos_corr_prufa.xlsx")
 



#Make image with significant correlations

allenAtlasImg = nib.load('allen_parcellation_zerbi165_subMASKED_G1_Left.nii.gz')
allenAtlas = allenAtlasImg.get_fdata()

#store dimensions of ROI
allenAtlasDim = allenAtlas.shape
allenAtlasDim
nVoxels = np.prod(allenAtlasDim)
nVoxels

# Reshape roi into a vector of size nVoxels
allenAtlas = np.reshape(allenAtlas,(nVoxels))	


# Find the indices inside roi
allenIndices = np.where(allenAtlas>0)

allenTeljari = allenIndices[0]
lengd = np.size(allenIndices)

significantIndices = np.zeros(lengd)

#Iterate through all areas in ROI where label >0, if label is equal to significantly correlated ROI change label into average correlation
#Note:  allenAtlas[allenTeljari[m]] == listiCA2[i] != 0.0 because allenAtlas value with the indices in allenTeljari is by definition >0



m=0
while m < lengd:
    i=0
    while i < staerd:
        if allenAtlas[allenTeljari[m]] == listi_G1_49[i]:
            heiltala = int(listi_G1_49[i]-1)
            significantIndices[m] = significant_G1_49[heiltala]
            break
        else:
            significantIndices[m] = 0
        i+=1
    m+=1

significantIndices
significantIndices = significantIndices.reshape(lengd,1)      
    
    
outfile = 'KFC' + "/" + 'allen_atlas_corr_G1_49_L' + ".nii.gz"

yDat = np.zeros(shape=allenAtlasDim + (1,))
yDat = np.reshape(yDat, (np.prod(allenAtlasDim), 1))
yDat[allenIndices,:] = significantIndices
yDat = np.reshape(yDat,allenAtlasDim)
yImg = nib.Nifti1Image(yDat,allenAtlasImg.get_affine(),allenAtlasImg.get_header())


nib.save(yImg,outfile)

    


