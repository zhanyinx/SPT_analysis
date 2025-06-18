nextflow.enable.dsl=2

include { DETECT_SPOTS } from './modules/detect_spots.nf'
include { TRACK_ASSIGN } from './modules/track_assign.nf'
include { MOTION_CORRECTION } from './modules/motion_correction.nf'
include { MSD_CALC } from './modules/msd_calc.nf'
include { DIRECTION_CALC } from './modules/direction_calc.nf'

workflow {
    Channel.fromPath("${params.input}/*.tif").set { tiff_files }

    DETECT_SPOTS(tiff_files)
    TRACK_ASSIGN(DETECT_SPOTS.out.xml)
    MOTION_CORRECTION(TRACK_ASSIGN.out.csv)
    MSD_CALC(MOTION_CORRECTION.out.corrected_dir)
    DIRECTION_CALC(MOTION_CORRECTION.out.corrected_dir)
}
