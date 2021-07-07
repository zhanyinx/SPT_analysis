#!/bin/bash
    
## Author(s): Yinxiu Zhan
## Contact: yinxiu.zhan@fmi.ch
## This software is distributed without any guarantee under the terms of the GNU General
## Public License, either Version 2, June 1991 or Version 3, June 2007.


function usage {
    echo -e "usage : run_msd_calculation.sh -i INPUT -s PATHSPT [-l MIN_LENGTH] [-p MIN_POINTS] [-o OUTPUT] [-s SCRATCH] [-h]"
    echo -e "Use option -h|--help for more information"
}

function help {
    usage;
    echo 
    echo "Run msd calculation."
    echo "---------------"
    echo "OPTIONS"
    echo
    echo "   -i|--input INPUT : input folder containing the *_corrected.csv files from TrackMate or motion correction."
    echo "   -s|--scriptpath SCRIPTPATH : path to SPT analysis github folder path"
    echo "   [-l|--min_length MIN_LENGTH] : Minimum number of timepoints per trajectory, default 10."
    echo "   [-p|--min_points MIN_POINTS] : Minimum number of points to calculate tamsd, default 5."
    echo "   [-o|--output OUTPUT] : output csv name, default output.csv"
    echo "   [-t|--tmp TMP] : scratch folder for temporary file, default ./scratch"
    echo "   [-h|--help]: help"
    exit;
}


# Transform long options to short ones
for arg in "$@"; do
  shift
  case "$arg" in
      "--input") set -- "$@" "-i" ;;
      "--scriptpath")   set -- "$@" "-s" ;;
      "--min_length")   set -- "$@" "-l" ;;
      "--min_points")   set -- "$@" "-p" ;;
      "--output")   set -- "$@" "-o" ;;
      "--tmp")   set -- "$@" "-t" ;;
       *)        set -- "$@" "$arg"
  esac
done

input=""
pathSPT=""
min_length=10
min_points=5
output="output.csv"
tmp="./scratch"

while getopts ":i:s:l:p:o:t:h" OPT
do
    case $OPT in
        i) input=$OPTARG;;
        s) pathSPT=$OPTARG;;
        l) min_length=$OPTARG;;
        p) min_points=$OPTARG;;
        o) output=$OPTARG;;
        t) tmp=$OPTARG;;
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

python $pathSPT/source/msd_calculation/msd.py -i $input -ml $min_length -mp $min_points -o $output -t $tmp
