process MOTION_CORRECTION {
    tag 'motion_correction'
    input:
        path csvs
    output:
        path 'corrected', emit: corrected_dir
    script:
        """
        mkdir -p uncorrected corrected
        mv ${csvs} uncorrected/
        python ${params.spt_path}/source/spot_detection_tracking/rototranslation_correction.py -i uncorrected -o corrected -t ${params.tmp}
        """
}
