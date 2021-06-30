#!/bin/bash
    
## Author(s): Yinxiu Zhan
## Contact: yinxiu.zhant@fmi.ch
## This software is distributed without any guarantee under the terms of the GNU General
## Public License, either Version 2, June 1991 or Version 3, June 2007.


function usage {
    echo -e "usage : 00_run_detection.sh -i INPUT -s PATHSPT [-n NGPU] [-h]"
    echo -e "Use option -h|--help for more information"
}

function help {
    usage;
    echo 
    echo "Detect spots using deepBlink."
    echo "See https://github.com/BBQuercus/deepBlink for details about deepBlink"
    echo "---------------"
    echo "OPTIONS"
    echo
    echo "   -i|--input INPUT : input folder containing the tiff file of all movies you want to run spot detection."
    echo "   -s|--scriptpath SCRIPTPATH : path to SPT analysis github folder path"
    echo "   [-n|--ngpu NGPU] : number of GPUs to use, default 1"
    echo "   [-h|--help]: help"
    exit;
}


# Transform long options to short ones
for arg in "$@"; do
  shift
  case "$arg" in
      "--input") set -- "$@" "-i" ;;
      "--ngpu")   set -- "$@" "-n" ;;
      "--scriptpath")   set -- "$@" "-s" ;;
      "--help")   set -- "$@" "-h" ;;
       *)        set -- "$@" "$arg"
  esac
done

pathtiff=""
pathSPT=""
ngpu=1

while getopts ":i:n:s:h" OPT
do
    case $OPT in
        i) pathtiff=$OPTARG;;
        n) ngpu=$OPTARG;;
        s) pathSPT=$OPTARG;;
        h) help ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            exit 1
            ;;
    esac
done

if [ $# -lt 4 ]
then
    usage
    exit
fi

if ! [ -d $pathtiff ]; then
    echo "$pathtiff does not exist!"
    exit
fi

if ! [ -d $pathSPT ]; then
    echo "$pathSPT does not exist!"
    exit
fi

gpu=0

for files in `ls $pathtiff/*tif`;do

	export CUDA_VISIBLE_DEVICES=$gpu
	python $pathSPT/source/spot_detection_tracking/3d_spot_detection.py -i $files -m $pathSPT/models/210518_141754_real_add_all_pia_180521.h5 -su "um" -pw 0.16 -hw 0.16 -vd 0.3 -ti 10 > nohup.out 2>&1 &
	let gpu=gpu+1
	if [ $gpu == $ngpu ]; then
		gpu=0
		wait
	fi
	
done
