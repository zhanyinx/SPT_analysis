process DIRECTION_CALC {
    tag 'direction'
    input:
        path dir
    output:
        path params.dir_output
    script:
        """
        python ${params.spt_path}/source/directionality_calculations/directions.py -i $dir -ml ${params.msd_min_length} -o ${params.dir_output} -t ${params.tmp}
        """
}
