#!/bin/bash
    
## Author(s): Yinxiu Zhan
## Contact: yinxiu.zhan@fmi.ch
## This software is distributed without any guarantee under the terms of the GNU General
## Public License, either Version 2, June 1991 or Version 3, June 2007.


function usage {
    echo -e "usage : run_msd_calculation.sh -i INPUT -s PATHSPT [-l MIN_LENGTH] [-p MIN_POINTS] [-o OUTPUT] [-s SCRATCH] [-u] [-h]"
    echo -e "Use option -h|--help for more information"
}

function help {
    usage;
    echo 
    echo "Run msd calculation."
    echo "---------------"
    echo "OPTIONS"
    echo
    echo "   -i|--input INPUT : input folder or single file containing the *_corrected.csv files from TrackMate or motion correction."
    echo "   -s|--scriptpath SCRIPTPATH : path to SPT analysis github folder path"
    echo "   [-l|--min_length MIN_LENGTH] : Minimum number of timepoints per trajectory, default 10."
    echo "   [-m|--min_tracks MIN_TRACKS] : Minimum number of tracks per cell. Cells with fewer tracks are filtered out."
    echo "   [-o|--output OUTPUT] : output csv name, default output.csv"
    echo "   [-p|--min_points MIN_POINTS] : Minimum number of points to calculate tamsd, default 5."
    echo "   [-t|--tmp TMP] : scratch folder for temporary file, default ./scratch"
    echo "   [-u|--uncorrected_residual]: if defined, it will look for *_uncorrected.csv and *_residual.csv files and output them in the results."
    echo "   [-w|--pairwise]: if defined - calculate pairwise MSD, otherwise regular MSD (based on single trajectories)."
    echo "   [-h|--help]: help"
    exit;
}


# Transform long options to short ones
for arg in "$@"; do
  shift
  case "$arg" in
      "--input") set -- "$@" "-i" ;;
      "--min_length")   set -- "$@" "-l" ;;
      "--min_points")   set -- "$@" "-p" ;;
      "--min_tracks")   set -- "$@" "-m" ;;
      "--output")   set -- "$@" "-o" ;;
      "--scriptpath")   set -- "$@" "-s" ;;
      "--tmp")   set -- "$@" "-t" ;;
      "--uncorrected_residual")   set -- "$@" "-u" ;;
      "--pairwise")   set -- "$@" "-w" ;;
       *)        set -- "$@" "$arg"
  esac
done

input=""
pathSPT=""
min_length=10
min_points=5
min_tracks=2
output="output.csv"
tmp="./scratch"
uncorrected_residual=1
pairwise=1
while getopts ":i:s:l:m:p:o:t:u:pwh" OPT
do
    case $OPT in
        i) input=$OPTARG;;
        l) min_length=$OPTARG;;
	m) min_tracks=$OPTARG;;
        o) output=$OPTARG;;
        p) min_points=$OPTARG;;
        s) pathSPT=$OPTARG;;
        t) tmp=$OPTARG;;
        u) uncorrected_residual=0;;
        w) pairwise=0;;
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

if ! [ -d $input ]; then
    echo "$input does not exist!"
    exit
fi

if ! [ -d $pathSPT ]; then
    echo "$pathSPT does not exist!"
    exit
fi

if [ $uncorrected_residual -eq 1 ]; then
    python $pathSPT/source/msd_calculation/msd.py -i $input -ml $min_length -mp $min_points -o $output -t $tmp -mt $min_tracks
elif [ $pairwise -eq 1 ]; then
    python $pathSPT/source/msd_calculation/msd.py -i $input -ml $min_length -mp $min_points -o $output -t $tmp -mt $min_tracks -pw $pairwise
else
    python $pathSPT/source/msd_calculation/msd.py -i $input -ml $min_length -mp $min_points -o $output -t $tmp -ur -mt $min_tracks
fi


