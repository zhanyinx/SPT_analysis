path="PATH2Tiff"

gpu=0
ngpu=2

for files in `ls $path/*tif`;do

	export CUDA_VISIBLE_DEVICES=$gpu
	python ../../source/spot_detection_tracking/3d_spot_detection.py -i $files -m ../../models/210518_141754_real_add_all_pia_180521.h5 -su "um" -pw 0.16 -hw 0.16 -vd 0.3 -ti 10 > nohup.out 2>&1 &
	let gpu=gpu+1
	if [ $gpu == $ngpu ]; then
		gpu=0
		wait
	fi
	
done
