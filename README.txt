In this repository are the scripts and ROIs used for Multimodal Gradient Mapping of Rodent Hippocampus. There are also data to test the python scripts.
Below is information on the purpose of the scripts in the repository.

Author of scripts (excluding congrads method): Brynja Gunnarsdóttir

Note - Author is not an experienced programmer

Python version: 3.7.6

Python Package                     Version            
---------------------------------- -------------------
glob2                              0.7                
ipython                            7.12.0             
jupyter                            1.0.0              
matplotlib                         3.1.3              
netneurotools                      0.2.1+117.g55386eb                     
nibabel                            3.0.2              
numpy                              1.18.1             
pathlib2                           2.3.5
pandas                             0.25.3  
requests                           2.22.0             
scikit-build                       0.10.0             
scikit-image                       0.16.2             
scikit-learn                       0.22.1                         
seaborn                            0.10.0             
selenium                           3.141.0            
sklearn                            0.0                
spyder                             4.0.1              
statsmodels                        0.9.0    


Functional Connectivity Scripts:


Gradient construction - https://github.com/koenhaak/congrads
ROIs used - whole-brain mask: allen_brainmask_SUBMASKED.nii.gz, hippocampus masks: allen_hippocampus_nosub_LEFT/RIGHT_1mm_erode_RS.nii.gz

Consensus clustering of gradients and functional connectivity (FC) analysis of clusters - kfc.py

Step-by-step method and scripts or methods used in each step:

1. Bag gradients from dataset into group-averaged gradient using whole dataset 500 times. Further cluster each group-averaged gradient using K-means algorithm
   into k = 2,3,...,10 clusters and save into matrix, B. Function used: BootstrapClusterG(voxelN, k, n, resampleSize, startMouse, endMouse, side, gradientNo), 
   voxelN = no. voxels within ROI, k = upper limit for clustering, n = no. of bagging rounds,resampleSize = size of resampled group, startMouse and 
   endMouse=numbered subjects range (e.g. whole dataset startMouse = 1, endMouse = 50), side = 'LEFT' or 'RIGHT', gradientNo = '1','2','3'....
   
2. Make consensus matrices from B. Function used: makeConsensusMat(mat), mat=string with numpy matrix name.

3. Choose optimal cluster solution by calculating proportion of ambiguous clustering (PAC) in the consensus matrices an Percent agreement and constructing 
   heatmaps. function used: PAC(mat) where mat is a string with the matrix name from makeConsensusMat function. Method for percent agreement found in kfc.py and 
   heatmaps constructed using sns.clustermap from seaborn.
   
4. Cluster consensus matrix with optimal cluster solution using spectral clustering using SpectralClustering from sklearn.cluster.

5. Conduct seed-based FC analysis for each cluster. The time-series from each cluster were correlated (Pearson R) to the ROIs of a resampled Allen Common
   Coordinate Framework (CCFv3)(V3, http://help.brain-map.org/download/attachments/2818169/MouseCCF.pdf). Done by following tutorial from 
   https://nilearn.github.io/auto_examples/03_connectivity/plot_signal_extraction.html. 
   
6. Find significant correlations were then found by conducting a non-parametric permutation test against zero for the correlations between each cluster and the ROIs
   for rest of the brain across the whole dataset and correcting for multiple comparisons using the Benjamini-Hochberg method with a false discovery rate (FDR) of
   0.005. Non-parametric permutation test conducted using nnstats.permtest_1samp from netneurotools, multiple comparison correction conducted using 
   statsmodels.stats.multitest.multipletests from statsmodels.


Hippocampus Gene Expression Gradients:


Method step-by-step and scripts or methods used in each step:

1. Download Allen Gene Expression Atlas (AGEA) correlation maps for all voxels within the ROI. Script - getHippoGenes.py

2. Transform correlation maps (.mhd files) into our reference space (QBI). Method implemented and conducted by Valerio Zerbi.

3. Find the seed voxel of each map after transformation. Necessary for similarity matrix construction and for building a hippocampus mask covered by all seed voxels within the ROI. Done by isolating the seed voxel of each map and transforming into our reference space, using fsl and the same method as in step 2, and retaining the voxel with the highest value after transformation, using fsl. 

Note - in this step the coordinates of the transformed seed voxel are entered into the filename of the transformed seed voxel and correlation maps. 

4. Create a ROI mask using seed voxels within the ROI. Script used - geneMaskMaker.sh. Fsl was then used to restrict the mask further by multiplying with the ROI mask in the FC gradient analysis.

5. Construct the similarity matrices for the ROI. Script used - geneSimilarityMatrixMaker.py

6. Enter similarity matrices into the Laplacian eigenmaps algorithm of the connectopic mapping method for gradient construction. Script used - adapted conmap.py from congrads method (https://github.com/koenhaak/congrads). Adaptation consisted only of changing variable names to fit with geneSimilarityMatrixMaker.py variable names. ROIs used - whole-brain mask: allen_brainmask_SUBMASKED.nii.gz, hippocampus masks: geneFC_Mask_LEFT/RIGHT.nii.gz
