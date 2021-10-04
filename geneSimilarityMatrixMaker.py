#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:59:54 2020

This script is how the gene expression similarity matrix is built before
constructing gradients with the congrads method (https://github.com/koenhaak/congrads) 
by computing the graph Laplacian of the ROI. 

@author: brynjagunnarsdottir
"""

import bisect
import glob
import numpy as np
import nibabel as nib
import sys
import errno
import matplotlib.pyplot as plt


def find_index(elements, value):
    index = bisect.bisect_left(elements, value)
    if index < len(elements) and elements[index] == value:
        return index
    

#Load ROI mask for Right hemisphere
    
imgMaskRight = nib.load('/path/to/right/ROI/mask.nii.gz')
dataMaskRight = imgMaskRight.get_fdata()
rightRoidims = dataMaskRight.shape
nVoxelsRight = np.prod(rightRoidims)
roiRight = np.reshape(dataMaskRight,(nVoxelsRight))	


# Find the indices inside roi

roiIndicesRight = np.where(dataMaskRight>0)

#Indices for Nifti image construction
#Only used to substitute "roiIndices" in conmap.py for image construction:
#yDat[roiIndices,:] = y[:,1:nmaps+1]

roiIndicesRightImg = np.where(roiRight>0)

#Array for indices for x-axis, y-axis and z-axis 

xArrR = roiIndicesRight[0] 

yArrR = roiIndicesRight[1]

zArrR = roiIndicesRight[2]

#Initiate Loop through xArrR,yArrR,zArrR to save indices as 6 numbers into array indexForSArrayRight, which 
#is used to insert gene maps into right location into similarity matrix

x=0

indexForSRight = []

while x < roiIndicesRight[0].size:
    stringIndex = (str(xArrR[x]).zfill(2)+str(yArrR[x]).zfill(2)+str(zArrR[x]).zfill(2))
    indexForSRight.append(int(stringIndex))
    x+=1
    
indexForSArrayRight = np.array(indexForSRight)
indexForSArrayRight = indexForSArrayRight.astype(np.int)

#Check that the list of strings indexForSRight is ordered, important for algorithm efficiency for finding
#correct location in S


for i in range(len(indexForSRight )-1):
    if indexForSRight [i] > indexForSRight [i+1]:
        print(i)
        result = False
        break
	else:
    	result = True
result

#Load mask for Left hemisphere
    
imgMaskLeft = nib.load('/path/to/left/ROI/mask.nii.gz')
dataMaskLeft = imgMaskLeft.get_fdata()
leftRoidims = dataMaskLeft.shape
nVoxelsLeft = np.prod(leftRoidims)
roiLeft = np.reshape(dataMaskLeft,(nVoxelsLeft))	

# Find the indices inside roi

roiIndicesLeft = np.where(dataMaskLeft>0)
roiIndicesLeft[0].size

#Indices for Nifti image
#Only used to substitute "roiIndices" in conmap.py for image construction:
#yDat[roiIndices,:] = y[:,1:nmaps+1]

roiIndicesLeftImg = np.where(roiLeft>0)

#Array for indices for x-axis, y-axis and z-axis 

xArrL = roiIndicesLeft[0] 

yArrL = roiIndicesLeft[1]

zArrL = roiIndicesLeft[2]

#Initiate Loop through xArrL,yArrL,zArrL to save indices as 6 numbers into array indexForSArrayLeft, which 
#is used to insert gene maps into left location into similarity matrix

x=0

indexForSLeft = []

while x < roiIndicesLeft[0].size:
    stringIndex = (str(xArrL[x]).zfill(2)+str(yArrL[x]).zfill(2)+str(zArrL[x]).zfill(2))
    indexForSLeft.append(int(stringIndex))
    x+=1
    
indexForSArrayLeft = np.array(indexForSLeft)
indexForSArrayLeft = indexForSArrayLeft.astype(np.int)

#Check that the list of strings indexForSLeft is ordered, important for algorithm efficiency for finding
#correct location in S


for i in range(len(indexForSLeft )-1):
    if indexForSLeft [i] > indexForSLeft [i+1]:
        print(i)
        result = False
        break
	else:
    	result = True
result


	
# Initialise similarity matrices

SRight = np.zeros([np.sum(roiRight>0),np.sum(roiRight>0)])

SLeft = np.zeros([np.sum(roiLeft>0),np.sum(roiLeft>0)])

#Go through all normalised correlation maps from AGEA and find whether each map belongs in Left or right hemisphere and which row of
#similarity matrix
#Note - the x-,y- and z-coordinates of the maps' seed voxel need to be in the filename of the correlation maps at the "indexCode"
#location of the filename for the loop to work.

#Keep count of how many maps go into Similarity matrices and how many do not

leftOut = 0
keptIn = 0

for name in glob.glob('/path/to/normalised/correlation/maps/'):
    
    #Right hemisphere x < 25, Left x > 25
    rightSide = (int(name[123:125]) < 25)
    indexCode = int(name[123:129])
    dataImg = nib.load(name)
    data = dataImg.get_fdata()
    
    #If parent voxel of map belongs to right or left side and falls within hippocampus mask, enter map into
    #correct row within similarity matrix. Else, do nothing and keep count of how many maps are left out
    
    if(rightSide):
        dataArray = data[roiIndicesRight]
        indexInS = find_index(indexForSArrayRight, indexCode)
        if indexInS is None:
            leftOut+=1
        else:
            print('Right '+str(indexInS))
            keptIn+=1 
            
            #If there is no input in SRight[indexInS,:] enter map 
            
            if SRight[indexInS,0] == 0.00:
                x=0
                while x < SRight[0,:].size:
                    SRight[indexInS,x] = dataArray[x]
                    x+=1
            #If there is input in SRight[indexinS], taka average of old and new input                               
                    
            else:
                x=0
                while x < SRight[0,:].size:
                    temp = (SRight[indexInS,x] + dataArray[x])/2.0
                    SRight[indexInS,x] = temp
                    x+=1
    else:
        dataArray = data[roiIndicesLeft]
        indexInS = find_index(indexForSArrayLeft, indexCode)
        if indexInS is None:
            leftOut+=1
        else:
            
            print('Left '+ str(indexInS))
            keptIn+=1 
            
            #If there is no input in SLeft[indexInS,:] enter map 
            
            if SLeft[indexInS,0] == 0.00:
                x=0
                while x < SLeft[0,:].size:
                    SLeft[indexInS,x] = dataArray[x]
                    x+=1
            #If there is input in SLeft[indexinS], taka average of old and new input                               
                    
            else:
                x=0
                while x < SLeft[0,:].size:
                    temp = (SLeft[indexInS,x] + dataArray[x])/2.0
                    SLeft[indexInS,x] = temp
                    x+=1
  

#Adjust similarity matrices so they are symmetric

np.fill_diagonal(SRight, 1.0)
SRightAdjusted = (SRight+SRight.transpose())/2 

np.fill_diagonal(SLeft, 1.0)
SLeftAdjusted = (SLeft+SLeft.transpose())/2 


#Next the graph Laplacian of the similarity matrices is computed using the conmap.py script of the congrads
#method exactly, only substituting the names of the similarity matrices etc.
