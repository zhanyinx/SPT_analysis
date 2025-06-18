process MSD_CALC {
    tag 'msd'
    input:
        path dir
    output:
        path params.msd_output
    script:
        """
        python ${params.spt_path}/source/msd_calculation/msd.py -i $dir -ml ${params.msd_min_length} -mp ${params.msd_min_points} -mt ${params.msd_min_tracks} -o ${params.msd_output} -t ${params.tmp}
        """
}
