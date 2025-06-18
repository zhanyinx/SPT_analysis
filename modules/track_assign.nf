process TRACK_ASSIGN {
    tag { xml.baseName }
    input:
        path xml
    output:
        path("tracks_gap_with_cellid/${xml.baseName}.tracks.with_cellIDs_uncorrected.csv"), emit: csv
        path("tracks_gap_with_cellid/${xml.baseName}.tracks.with_cellIDs.xml"), emit: tracked_xml
    script:
        """
        mkdir -p tracks_gap tracks_gap_with_cellid
        ${params.fiji} --ij2 --headless --run ${params.spt_path}/source/spot_detection_tracking/trackmate_tracking_gap.py \
            'basedir="$PWD/tracks_gap",xml="$xml",link_dist="${params.link_dist}",gap_closing_dist="${params.gap_closing_dist}",ngap_max="${params.ngap_max}"'
        mask="${params.mask_dir}/${xml.baseName}"
        ${params.fiji} --ij2 --headless --run ${params.spt_path}/source/spot_detection_tracking/assign_cellids.py \
            'basedir="$PWD/tracks_gap_with_cellid",xml="$PWD/tracks_gap/${xml.baseName}.tracks.xml",labeledFrames="${params.labeled_frames}",maskfile="$mask"'
        """
}
