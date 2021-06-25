#!/bin/bash
if ! [ -d tracks_gap ]; then
        mkdir tracks_gap
fi

rm apporun.sh

for file in `ls um_based/*xml`; do


	echo "nice -n 19 /tungstenfs/scratch/ggiorget/Fiji.app/ImageJ-linux64 --ij2 --headless --run ../trackmate_tracking_gap.py 'basedir=\"/tungstenfs/scratch/ggiorget/zhan/2021/1105_pia_image_analysis/3d_prediction/Rad21_cell_lines/tracks_gap\",xml=\"$file\"'" >> apporun.sh

done

sh apporun.sh
rm apporun.sh

if ! [ -d tracks_gap_with_cellid ]; then
	mkdir tracks_gap_with_cellid
fi


for file in `ls tracks_gap/*xml`; do

	name=`echo $file | xargs -0 -n 1 basename | sed 's/\.tracks\.xml//g'`
	
	mask=`ls ../../trackmate_manual_analysis/renamed_masks/$name`
	echo "nice -n 19 /tungstenfs/scratch/ggiorget/Fiji.app/ImageJ-linux64 --ij2 --headless --run ../assign_cellids.py 'basedir=\"./tracks_gap_with_cellid\",xml=\"$file\",labeledFrames=\"93\",maskfile=\"$mask\"'" >> apporun1.sh

done

sh apporun1.sh
rm apporun1.sh
