process DETECT_SPOTS {
    tag { tiff.baseName }
    input:
        path tiff
    output:
        path("${tiff.baseName}.xml"), emit: xml
        path("${tiff.baseName}.pdf"), emit: pdf
    script:
        def script_path = params.mode == '2d' ?
            "${params.spt_path}/source/spot_detection_tracking/2d_spot_detection.py" :
            "${params.spt_path}/source/spot_detection_tracking/3d_spot_detection.py"
        def extra = params.mode == '3d' ? "-vd ${params.voxeldepth}" : ''
        def model = "${params.spt_path}/models/210518_141754_real_add_all_pia_180521.h5"
        """
        python ${script_path} -i $tiff -m ${model} -su um -pw ${params.pixelwidth} \
            -hw ${params.pixelheight} -ti ${params.timeres} ${extra} -o ${tiff.baseName}.pdf
        mv pixel_based/${tiff.baseName}.xml .
        """
}
