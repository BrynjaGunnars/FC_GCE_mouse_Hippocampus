#!/bin/bash

#Script to make mask (non-binarized!) using AGEA 'seed' voxels, need isolated and normalized seed voxels and empty mask file, made
#by e.g. zero-ing out one normalized AGEA map using fslmaths

#For the test run to work, the user needs to download the files on the directory CA2 on
#https://github.com/BrynjaGunnars/FC_GCE_mouse_Hippocampus/tree/main/test_data and maintain and
#set the working directory to CA2


find . -type f -name "*swap_QBI_QBIparent*"| \
while read file ; do
    echo "processing ${file}" 
    OldTimestamp="$(date -r empty_mask.nii.gz)"
    fslmaths empty_mask.nii.gz -add "${file}" empty_mask.nii.gz
    NewTimestamp=$OldTimestamp
	while [ "$NewTimestamp" = "$OldTimestamp" ];  do 
   		sleep 1 
  		NewTimestamp="$(date -r empty_mask.nii.gz)"
   		echo "old: $OldTimestamp"; echo "new: $NewTimestamp"
	done
done
