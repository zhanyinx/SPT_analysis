#!/bin/bash

if ! [ -d tracks ]; then
	mkdir tracks
fi

rm apporun.sh

for file in `ls um_based/*xml`; do


	echo "nice -n 19 /tungstenfs/scratch/ggiorget/Fiji.app/ImageJ-linux64 --ij2 --headless --run ../trackmate_tracking.py 'basedir=\"/tungstenfs/scratch/ggiorget/zhan/2021/1105_pia_image_analysis/3d_prediction/Rad21_cell_lines/tracks\",xml=\"$file\"'" >> apporun.sh

done

sh apporun.sh
rm apporun.sh
