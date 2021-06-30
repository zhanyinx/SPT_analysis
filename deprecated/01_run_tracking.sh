#!/bin/bash
    
## Author(s): Yinxiu Zhan
## Contact: yinxiu.zhant@fmi.ch
## This software is distributed without any guarantee under the terms of the GNU General
## Public License, either Version 2, June 1991 or Version 3, June 2007.


function usage {
    echo -e "usage : 01_run_tracking.sh -i INPUT -s PATHSPT -f PATHFIJI [-h]"
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
    echo "   -s|--scriptpath SCRIPTPATH : path to SPT analysis github folder path"
    echo "   -f|--fijipath FIJIPATH : path to executable fiji (ImageJ-...64)"
    echo "   [-h|--help]: help"
    exit;
}


# Transform long options to short ones
for arg in "$@"; do
  shift
  case "$arg" in
      "--input") set -- "$@" "-i" ;;
      "--f")   set -- "$@" "-f" ;;
      "--scriptpath")   set -- "$@" "-s" ;;
      "--help")   set -- "$@" "-h" ;;
       *)        set -- "$@" "$arg"
  esac
done

pathxml=""
pathSPT=""
pathfiji=""

while getopts ":i:f:s:h" OPT
do
    case $OPT in
        i) pathxml=$OPTARG;;
        f) pathfiji=$OPTARG;;
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

if [ $# -lt 6 ]
then
    usage
    exit
fi

if ! [ -d $pathxml ]; then
    echo "$pathxml does not exist!"
    exit
fi

if ! [ -d $pathSPT ]; then
    echo "$pathSPT does not exist!"
    exit
fi

if ! [ -d $pathfiji ]; then
    echo "$pathfiji does not exist!"
    exit
fi

if ! [ -d tracks ]; then
    mkdir tracks
fi

if [ -f apporun.sh ]; then
    rm apporun.sh
fi

basedir=`pwd`

for file in `ls $input/*xml`; do


	echo "nice -n 19 $pathfiji --ij2 --headless --run $pathSPT/source/spot_detection_tracking/trackmate_tracking.py 'basedir=\"$basedir/tracks\",xml=\"$file\"'" >> apporun.sh

done

sh apporun.sh
rm apporun.sh
