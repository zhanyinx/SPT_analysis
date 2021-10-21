#!/bin/bash
    
## Author(s): Yinxiu Zhan
## Contact: yinxiu.zhan@fmi.ch
## This software is distributed without any guarantee under the terms of the GNU General
## Public License, either Version 2, June 1991 or Version 3, June 2007.


function usage {
    echo -e "usage : 01_tracking_cell_assignments_with_gaps.sh -i INPUT -s PATHSPT -f PATHFIJI -m PATHMASK [-l LINK_DIST] [-g GAP_CLOSING_DIST] [-n NGAP_MAX] [-h]"
    echo -e "Use option -h|--help for more information"
}

function help {
    usage;
    echo 
    echo "Tracking and cell assignment."
    echo "---------------"
    echo "OPTIONS"
    echo
    echo "   -i|--input INPUT : input folder containing the xml file of all movies you want to run spot detection."
    echo "   -s|--scriptpath SCRIPTPATH : path to spt analysis github folder path"
    echo "   -f|--fijipath FIJIPATH : path to executable fiji (ImageJ-...64)"
    echo "   -m|--maskpath MASKPATH : path to masks used for cell id assignment"
    echo "   [-l|--link_dist LINK_DIST] : linking distance for TrackMate"
    echo "   [-g|--gap_closing_dist GAP_CLOSING_DIST] : gap closing distance for TrackMate"
    echo "   [-n|--ngap_max NGAP_MAX] : maximum number of gaps allowed in TrackMate"
    echo "   [-h|--help]: help"
    exit;
}


# Transform long options to short ones
for arg in "$@"; do
  shift
  case "$arg" in
      "--input") set -- "$@" "-i" ;;
      "--fijipath")   set -- "$@" "-f" ;;
      "--ngap_max")   set -- "$@" "-n" ;;
      "--link_dist")   set -- "$@" "-l" ;;
      "--gap_closing_dist")   set -- "$@" "-g" ;;
      "--maskpath")   set -- "$@" "-m" ;;
      "--scriptpath")   set -- "$@" "-s" ;;
      "--help")   set -- "$@" "-h" ;;
       *)        set -- "$@" "$arg"
  esac
done

pathxml=""
pathspt=""
pathfiji=""
pathmask=""
link_dist=0.8
gap_closing_dist=0.8
ngap_max=1

while getopts ":i:m:f:s:l:g:n:h" OPT
do
    case $OPT in
        i) pathxml=$OPTARG;;
        f) pathfiji=$OPTARG;;
        s) pathspt=$OPTARG;;
        l) link_dist=$OPTARG;;
        g) gap_closing_dist=$OPTARG;;
        n) ngap_max=$OPTARG;;
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
    echo "nice -n 19 $pathfiji --ij2 --headless --run $pathspt/source/spot_detection_tracking/trackmate_tracking_gap.py 'basedir=\"$basedir/tracks_gap\",xml=\"$file\",link_dist=\"$link_dist\",gap_closing_dist=\"$gap_closing_dist\",ngap_max=\"$ngap_max\"'" >> apporun.sh
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
