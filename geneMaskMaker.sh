#!/bin/bash
#Script to make mask (non-binarized!) using AGEA 'seed' voxels, need isolated and normalized seed voxels and empty mask file, made
#by e.g. zero-ing out one normalized AGEA map using fslmaths

find "/path/to/downloaded/AGEA/data/directory" -type f -name "name to specify seed voxels"| \
while read file ; do
    echo "processing ${file}" 
    OldTimestamp="$(date -r path/to/mask/filename)"
    fslmaths path/to/mask/filename -add "${file}" path/to/mask/filename
    NewTimestamp=$OldTimestamp
	while [ "$NewTimestamp" = "$OldTimestamp" ];  do 
   		sleep 1 
  		NewTimestamp="$(date -r path/to/mask/filename)"
   		echo "old: $OldTimestamp"; echo "new: $NewTimestamp"
	done
done