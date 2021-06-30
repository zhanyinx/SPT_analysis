#!/bin/bash

if ! [ -d tracks_with_cellid ]; then
	mkdir tracks_with_cellid
fi


for file in `ls tracks/*xml`; do

	name=`echo $file | xargs -0 -n 1 basename | sed 's/\.tracks\.xml//g'`
	
	mask=`ls ../../trackmate_manual_analysis/renamed_masks/$name`
	echo "nice -n 19 /tungstenfs/scratch/ggiorget/Fiji.app/ImageJ-linux64 --ij2 --headless --run ../assign_cellids.py 'basedir=\"./tracks_with_cellid\",xml=\"$file\",labeledFrames=\"93\",maskfile=\"$mask\"'" >> apporun1.sh

done

sh apporun1.sh
rm apporun1.sh
