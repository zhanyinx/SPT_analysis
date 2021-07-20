#!/bin/bash
    
## Author(s): Yinxiu Zhan, Pavel Kos
## Contact: yinxiu.zhan@fmi.ch
## This software is distributed without any guarantee under the terms of the GNU General
## Public License, either Version 2, June 1991 or Version 3, June 2007.


function usage {
    echo -e "usage : run_directions_calculation.sh -i INPUT -s PATHSPT [-l MIN_LENGTH] [-o OUTPUT] [-s SCRATCH] [-u] [-h]"
    echo -e "Use option -h|--help for more information"
}

function help {
    usage;
    echo 
    echo "Run angles calculation. Bonus: D and alpha for 3-points MSD."
    echo "---------------"
    echo "OPTIONS"
    echo
    echo "   -i|--input INPUT : input folder or single file containing the *_corrected.csv files from TrackMate or motion correction."
    echo "   -s|--scriptpath SCRIPTPATH : path to SPT analysis github folder path"
    echo "   [-l|--min_length MIN_LENGTH] : Minimum number of timepoints per trajectory, default 10."
    echo "   [-o|--output OUTPUT] : output csv name, default angle_d_alpha.csv"
    echo "   [-t|--tmp TMP] : scratch folder for temporary file, default ./scratch"
    echo "   [-u|--uncorrected_residual]: if defined, it will look for *_uncorrected.csv and *_residual.csv files and output them in the results."
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
      "--output")   set -- "$@" "-o" ;;
      "--uncorrected_residual")   set -- "$@" "-u" ;;
      "--tmp")   set -- "$@" "-t" ;;
       *)        set -- "$@" "$arg"
  esac
done

input=""
pathSPT=""
min_length=10
output="angle_d_alpha.csv"
tmp="./scratch"
uncorrected_residual=1

while getopts ":i:s:l:p:o:t:uh" OPT
do
    case $OPT in
        i) input=$OPTARG;;
        s) pathSPT=$OPTARG;;
        l) min_length=$OPTARG;;
        o) output=$OPTARG;;
        t) tmp=$OPTARG;;
        u) uncorrected_residual=0;;
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
    python $pathSPT/source/directionality_calculations/directions.py -i $input -ml $min_length -o $output -t $tmp
else
    python $pathSPT/source/directionality_calculations/directions.py -i $input -ml $min_length -o $output -t $tmp -ur
fi

