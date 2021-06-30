#!/bin/bash
    
## Author(s): Yinxiu Zhan
## Contact: yinxiu.zhant@fmi.ch
## This software is distributed without any guarantee under the terms of the GNU General
## Public License, either Version 2, June 1991 or Version 3, June 2007.


function usage {
    echo -e "usage : 01_tracking_cell_assignments_with_gaps.sh -i INPUT -s PATHSPT -f PATHFIJI -m PATHMASK [-h]"
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
    echo "   -i|--input INPUT : input folder containing the xml file of all movies you want to run spot detection."
    echo "   -s|--scriptpath SCRIPTPATH : path to spt analysis github folder path"
    echo "   -f|--fijipath FIJIPATH : path to executable fiji (ImageJ-...64)"
    echo "   -m|--maskpath MASKPATH : path to masks used for cell id assignment"
    echo "   [-h|--help]: help"
    exit;
}


# Transform long options to short ones
for arg in "$@"; do
  shift
  case "$arg" in
      "--input") set -- "$@" "-i" ;;
      "--fijipath")   set -- "$@" "-f" ;;
      "--scriptpath")   set -- "$@" "-s" ;;
      "--maskpath")   set -- "$@" "-m" ;;
      "--help")   set -- "$@" "-h" ;;
       *)        set -- "$@" "$arg"
  esac
done

pathxml=""
pathspt=""
pathfiji=""
pathmask=""

while getopts ":i:m:f:s:h" OPT
do
    case $OPT in
        i) pathxml=$OPTARG;;
        f) pathfiji=$OPTARG;;
        s) pathspt=$OPTARG;;
        m) pathmask=$OPTARG;;
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

if [ $# -lt 8 ]
then
    usage
    exit
fi

if ! [ -d $pathxml ]; then
    echo "$pathxml does not exist!"
    exit
fi

if ! [ -d $pathspt ]; then
    echo "$pathspt does not exist!"
    exit
fi

if ! [ -f $pathfiji ]; then
    echo "$pathfiji does not exist!"
    exit
fi

if ! [ -d $pathmask ]; then
    echo "$pathmask does not exist!"
    exit
fi

if [ -f apporun.sh ]; then
    rm apporun.sh
fi

if [ -f apporun1.sh ]; then
    rm apporun1.sh
fi


if ! [ -d tracks_gap ]; then
        mkdir tracks_gap
fi

basedir=`pwd`

for file in `ls $pathxml/*xml`; do
    echo "nice -n 19 $pathfiji --ij2 --headless --run $pathspt/source/spot_detection_tracking/trackmate_tracking_gap.py 'basedir=\"$basedir/tracks_gap\",xml=\"$file\"'" >> apporun.sh
done

sh apporun.sh
rm apporun.sh

if ! [ -d tracks_gap_with_cellid ]; then
	mkdir tracks_gap_with_cellid
fi


for file in `ls $basedir/tracks_gap/*xml`; do
	name=`echo $file | xargs -0 -n 1 basename | sed 's/\.tracks\.xml//g'`
	
	mask=`ls $pathmask/$name`
	echo "nice -n 19 $pathfiji --ij2 --headless --run $pathspt/source/spot_detection_tracking/assign_cellids.py 'basedir=\"$basedir/tracks_gap_with_cellid\",xml=\"$file\",labeledFrames=\"93\",maskfile=\"$mask\"'" >> apporun1.sh

done

sh apporun1.sh
rm apporun1.sh
