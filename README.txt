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
scipy                              1.2.0              
seaborn                            0.10.0             
selenium                           3.141.0            
sklearn                            0.0                
spyder                             4.0.1              
statsmodels                        0.9.0    


Functional Connectivity Scripts:


Gradient construction - https://github.com/koenhaak/congrads
ROIs used - whole-brain mask: allen_brainmask_SUBMASKED.nii.gz, hippocampus masks: allen_hippocampus_nosub_LEFT/RIGHT_1mm_erode_RS.nii.gz

K-means clustering of gradients and FC analysis of clusters - kfc.py


Hippocampus Gene Expression Gradients:


Method step-by-step and scripts or methods used in each step:

1. Download Allen Gene Expression Atlas (AGEA) correlation maps for all voxels within the ROI. Script - getHippoGenes.py

2. Transform correlation maps (.mhd files) into our reference space (QBI). Method implemented and conducted by Valerio Zerbi.

3. Find the seed voxel of each map after transformation. Necessary for similarity matrix construction and for building a hippocampus mask covered by all seed voxels within the ROI. Done by isolating the seed voxel of each map and transforming into our reference space, using fsl and the same method as in step 2, and retaining the voxel with the highest value after transformation, using fsl. 

Note - in this step the coordinates of the transformed seed voxel are entered into the filename of the transformed seed voxel and correlation maps. 

4. Create a ROI mask using seed voxels within the ROI. Script used - geneMaskMaker.sh. Fsl was then used to restrict the mask further by multiplying with the ROI mask in the FC gradient analysis.

5. Construct the similarity matrices for the ROI. Script used - geneSimilarityMatrixMaker.py

6. Enter similarity matrices into the Laplacian eigenmaps algorithm of the connectopic mapping method for gradient construction. Script used - adapted conmap.py from congrads method (https://github.com/koenhaak/congrads). Adaptation consisted only of changing variable names to fit with geneSimilarityMatrixMaker.py variable names. ROIs used - whole-brain mask: allen_brainmask_SUBMASKED.nii.gz, hippocampus masks: geneFC_Mask_LEFT/RIGHT.nii.gz
