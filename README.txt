In this repository are the scripts and ROIs used for Multimodal Gradient Mapping of Rodent Hippocampus, and for what purpose they were used.

Author of scripts (excluding congrads method): Brynja Gunnarsdóttir

Note - Author is not an experienced programmer


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
