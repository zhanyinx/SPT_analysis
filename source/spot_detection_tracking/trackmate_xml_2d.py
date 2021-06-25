"""\U0001F1EB\U0001F1EF  \U00002B50 CSV track coordinate to TrackMate XML conversion.
Fiji allows for quick and easy viewing of images. TrackMate can be used to view tracks.
Unfortunately, it isn't that simple to convert "normal" coordinate output into
TrackMate-viewable format.
Requires a "tracks.csv" file that contains the following columns:
- x, y: Coordinate positions in x-/y-axis
- particle: Unique ID assigned to all coordinates along one track
- frame: Current point in time / frame
"""

import argparse
import os
import tempfile
import xml.dom.minidom
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import skimage.io


def get_gaps(frames):
    def __longest_consecutive(a):
        """Return length of longest consecutive range in list of integers."""
        a = set(a)
        longest = 0
        for i in a:
            if i - 1 not in a:
                streak = 0
                while i in a:
                    i += 1
                    streak += 1
                    longest = max(longest, streak)
        return longest

    full_length = np.arange(min(frames), max(frames))
    diff = np.setdiff1d(full_length, frames)
    longest = __longest_consecutive(diff)
    total = len(diff)
    return str(longest), str(total), str(len(full_length))


def __create_model(root, spatialunits: str = "pixel", timeunits: str = "sec"):
    dict_spotfeatures = [
        {
            "feature": "QUALITY",
            "name": "Quality",
            "shortname": "Quality",
            "dimension": "QUALITY",
            "isint": "false",
        },
        {
            "feature": "POSITION_X",
            "name": "X",
            "shortname": "X",
            "dimension": "POSITION",
            "isint": "false",
        },
        {
            "feature": "POSITION_Y",
            "name": "Y",
            "shortname": "Y",
            "dimension": "POSITION",
            "isint": "false",
        },
        {
            "feature": "POSITION_Z",
            "name": "Z",
            "shortname": "Z",
            "dimension": "POSITION",
            "isint": "false",
        },
        {
            "feature": "POSITION_T",
            "name": "T",
            "shortname": "T",
            "dimension": "TIME",
            "isint": "false",
        },
        {
            "feature": "FRAME",
            "name": "Frame",
            "shortname": "Frame",
            "dimension": "NONE",
            "isint": "true",
        },
        {
            "feature": "RADIUS",
            "name": "Radius",
            "shortname": "R",
            "dimension": "LENGTH",
            "isint": "false",
        },
        {
            "feature": "VISIBILITY",
            "name": "Visibility",
            "shortname": "Visibility",
            "dimension": "NONE",
            "isint": "true",
        },
        {
            "feature": "MANUAL_INTEGER_SPOT_FEATURE",
            "name": "Custom Integer Spot Feature",
            "shortname": "Integer Spot Feature",
            "dimension": "NONE",
            "isint": "true",
        },
        {
            "feature": "MANUAL_DOUBLE_SPOT_FEATURE",
            "name": "Custom Double Spot Feature",
            "shortname": "Double Spot Feature",
            "dimension": "NONE",
            "isint": "false",
        },
        {
            "feature": "HAS_MAX_QUALITY_IN_FRAME",
            "name": "Has max quality",
            "shortname": "Max Quality",
            "dimension": "NONE",
            "isint": "true",
        },
        {
            "feature": "MANUAL_COLOR",
            "name": "Manual spot color",
            "shortname": "Spot color",
            "dimension": "NONE",
            "isint": "true",
        },
        {
            "feature": "MEAN_INTENSITY",
            "name": "Mean intensity",
            "shortname": "Mean",
            "dimension": "INTENSITY",
            "isint": "false",
        },
        {
            "feature": "MEDIAN_INTENSITY",
            "name": "Median intensity",
            "shortname": "Median",
            "dimension": "INTENSITY",
            "isint": "false",
        },
        {
            "feature": "MIN_INTENSITY",
            "name": "Minimal intensity",
            "shortname": "Min",
            "dimension": "INTENSITY",
            "isint": "false",
        },
        {
            "feature": "MAX_INTENSITY",
            "name": "Maximal intensity",
            "shortname": "Max",
            "dimension": "INTENSITY",
            "isint": "false",
        },
        {
            "feature": "TOTAL_INTENSITY",
            "name": "Total intensity",
            "shortname": "Total int.",
            "dimension": "INTENSITY",
            "isint": "false",
        },
        {
            "feature": "STANDARD_DEVIATION",
            "name": "Standard deviation",
            "shortname": "Stdev.",
            "dimension": "INTENSITY",
            "isint": "false",
        },
        {
            "feature": "ESTIMATED_DIAMETER",
            "name": "Estimated diameter",
            "shortname": "Diam.",
            "dimension": "LENGTH",
            "isint": "false",
        },
        {
            "feature": "CONTRAST",
            "name": "Contrast",
            "shortname": "Constrast",
            "dimension": "NONE",
            "isint": "false",
        },
        {
            "feature": "SNR",
            "name": "Signal/Noise, ratio",
            "shortname": "SNR",
            "dimension": "NONE",
            "isint": "false",
        },
    ]

    dict_edgefeatures = [
        {
            "feature": "SPOT_SOURCE_ID",
            "name": "Source spot ID",
            "shortname": "Source ID",
            "dimension": "NONE",
            "isint": "true",
        },
        {
            "feature": "SPOT_TARGET_ID",
            "name": "Target spot ID",
            "shortname": "Target ID",
            "dimension": "NONE",
            "isint": "true",
        },
        {
            "feature": "LINK_COST",
            "name": "Link cost",
            "shortname": "Cost",
            "dimension": "NONE",
            "isint": "false",
        },
        {
            "feature": "EDGE_TIME",
            "name": "Time (mean)",
            "shortname": "T",
            "dimension": "TIME",
            "isint": "false",
        },
        {
            "feature": "EDGE_X_LOCATION",
            "name": "X Location (mean)",
            "shortname": "X",
            "dimension": "POSITION",
            "isint": "false",
        },
        {
            "feature": "EDGE_Y_LOCATION",
            "name": "Y Location (mean)",
            "shortname": "Y",
            "dimension": "POSITION",
            "isint": "false",
        },
        {
            "feature": "EDGE_Z_LOCATION",
            "name": "Z Location (mean)",
            "shortname": "Z",
            "dimension": "POSITION",
            "isint": "false",
        },
        {
            "feature": "VELOCITY",
            "name": "Velocity",
            "shortname": "V",
            "dimension": "VELOCITY",
            "isint": "false",
        },
        {
            "feature": "DISPLACEMENT",
            "name": "Displacement",
            "shortname": "D",
            "dimension": "LENGTH",
            "isint": "false",
        },
        {
            "feature": "MANUAL_COLOR",
            "name": "Manual edge color",
            "shortname": "Edge color",
            "dimension": "NONE",
            "isint": "true",
        },
    ]

    dict_trackfeatures = [
        {
            "feature": "MANUAL_INTEGER_TRACK_FEATURE",
            "name": "Custom Integer Track Feature",
            "shortname": "Integer Track Feature",
            "dimension": "NONE",
            "isint": "true",
        },
        {
            "feature": "MANUAL_DOUBLE_TRACK_FEATURE",
            "name": "Custom Double Track Feature",
            "shortname": "Double Track Feature",
            "dimension": "NONE",
            "isint": "false",
        },
        {
            "feature": "NUMBER_SPOTS",
            "name": "Number of spots in track",
            "shortname": "N spots",
            "dimension": "NONE",
            "isint": "true",
        },
        {
            "feature": "NUMBER_GAPS",
            "name": "Number of gaps",
            "shortname": "Gaps",
            "dimension": "NONE",
            "isint": "true",
        },
        {
            "feature": "LONGEST_GAP",
            "name": "Longest gap",
            "shortname": "Longest gap",
            "dimension": "NONE",
            "isint": "true",
        },
        {
            "feature": "NUMBER_SPLITS",
            "name": "Number of split events",
            "shortname": "Splits",
            "dimension": "NONE",
            "isint": "true",
        },
        {
            "feature": "NUMBER_MERGES",
            "name": "Number of merge events",
            "shortname": "Merges",
            "dimension": "NONE",
            "isint": "true",
        },
        {
            "feature": "NUMBER_COMPLEX",
            "name": "Complex points",
            "shortname": "Complex",
            "dimension": "NONE",
            "isint": "true",
        },
        {
            "feature": "TRACK_DURATION",
            "name": "Duration of track",
            "shortname": "Duration",
            "dimension": "TIME",
            "isint": "false",
        },
        {
            "feature": "TRACK_START",
            "name": "Track start",
            "shortname": "T start",
            "dimension": "TIME",
            "isint": "false",
        },
        {
            "feature": "TRACK_STOP",
            "name": "Track stop",
            "shortname": "T stop",
            "dimension": "TIME",
            "isint": "false",
        },
        {
            "feature": "TRACK_DISPLACEMENT",
            "name": "Track displacement",
            "shortname": "Displacement",
            "dimension": "LENGTH",
            "isint": "false",
        },
        {
            "feature": "TRACK_INDEX",
            "name": "Track index",
            "shortname": "Index",
            "dimension": "NONE",
            "isint": "true",
        },
        {
            "feature": "TRACK_ID",
            "name": "Track ID",
            "shortname": "ID",
            "dimension": "NONE",
            "isint": "true",
        },
        {
            "feature": "TRACK_X_LOCATION",
            "name": "X Location (mean)",
            "shortname": "X",
            "dimension": "POSITION",
            "isint": "false",
        },
        {
            "feature": "TRACK_Y_LOCATION",
            "name": "Y Location (mean)",
            "shortname": "Y",
            "dimension": "POSITION",
            "isint": "false",
        },
        {
            "feature": "TRACK_Z_LOCATION",
            "name": "Z Location (mean)",
            "shortname": "Z",
            "dimension": "POSITION",
            "isint": "false",
        },
        {
            "feature": "TRACK_MEAN_SPEED",
            "name": "Mean velocity",
            "shortname": "Mean V",
            "dimension": "VELOCITY",
            "isint": "false",
        },
        {
            "feature": "TRACK_MAX_SPEED",
            "name": "Maximal velocity",
            "shortname": "Max V",
            "dimension": "VELOCITY",
            "isint": "false",
        },
        {
            "feature": "TRACK_MIN_SPEED",
            "name": "Minimal velocity",
            "shortname": "Min V",
            "dimension": "VELOCITY",
            "isint": "false",
        },
        {
            "feature": "TRACK_MEDIAN_SPEED",
            "name": "Median velocity",
            "shortname": "Median V",
            "dimension": "VELOCITY",
            "isint": "false",
        },
        {
            "feature": "TRACK_STD_SPEED",
            "name": "Velocity standard deviation",
            "shortname": "V std",
            "dimension": "VELOCITY",
            "isint": "false",
        },
        {
            "feature": "TRACK_MEAN_QUALITY",
            "name": "Mean quality",
            "shortname": "Mean Q",
            "dimension": "QUALITY",
            "isint": "false",
        },
        {
            "feature": "TRACK_MAX_QUALITY",
            "name": "Maximal quality",
            "shortname": "Max Q",
            "dimension": "QUALITY",
            "isint": "false",
        },
        {
            "feature": "TRACK_MIN_QUALITY",
            "name": "Minimal quality",
            "shortname": "Min Q",
            "dimension": "QUALITY",
            "isint": "false",
        },
        {
            "feature": "TRACK_MEDIAN_QUALITY",
            "name": "Median quality",
            "shortname": "Median Q",
            "dimension": "QUALITY",
            "isint": "false",
        },
        {
            "feature": "TRACK_STD_QUALITY",
            "name": "Quality standard deviation",
            "shortname": "Q std",
            "dimension": "QUALITY",
            "isint": "false",
        },
    ]
    # Model
    model = ET.SubElement(root, "Model", spatialunits=spatialunits, timeunits=timeunits)
    featuredeclarations = ET.SubElement(model, "FeatureDeclarations")

    # SpotFeatures
    spotfeatures = ET.SubElement(featuredeclarations, "SpotFeatures")
    for dct in dict_spotfeatures:
        _ = ET.SubElement(spotfeatures, "Feature", **dct)

    # Edgefeatures
    edgefeatures = ET.SubElement(featuredeclarations, "EdgeFeatures")
    for dct in dict_edgefeatures:
        _ = ET.SubElement(edgefeatures, "Feature", **dct)

    # TrackFeatures
    trackfeatures = ET.SubElement(featuredeclarations, "TrackFeatures")
    for dct in dict_trackfeatures:
        _ = ET.SubElement(trackfeatures, "Feature", **dct)

    return model


def __create_allspots(model, df):
    # List of all spots (without tracks)
    allspots = ET.SubElement(model, "AllSpots", nspots=str(len(df)))
    spotid = 0
    for frame in df["slice"].unique():
        frame_id = str(float(frame))
        df_frame = df[df["slice"] == frame]
        spotsinframe = ET.SubElement(allspots, "SpotsInFrame", frame=str(frame))
        for row in df_frame.iterrows():
            try:
                size = str(row[1]["size"] * 2)
            except KeyError:
                size = "1.0"
            dict_spot = {
                "ID": f"{spotid:06}",
                "name": f"ID{spotid:06}",
                "QUALITY": "1.0",
                "POSITION_T": frame_id,
                "MAX_INTENSITY": "1.0",
                "FRAME": frame_id,
                "MEDIAN_INTENSITY": "1.0",
                "VISIBILITY": "1",
                "MEAN_INTENSITY": "1.0",
                "TOTAL_INTENSITY": "1.0",
                "ESTIMATED_DIAMETER": size,
                "RADIUS": "1.0",
                "SNR": "1.0",
                "POSITION_X": str(row[1]["x"]),
                "POSITION_Y": str(row[1]["y"]),
                "STANDARD_DEVIATION": "1.0",
                "CONTRAST": "1.0",
                "MANUAL_COLOR": "-10921639",
                "MIN_INTENSITY": "0.0",
                "POSITION_Z": "1",
            }
            _ = ET.SubElement(spotsinframe, "Spot", **dict_spot)
            spotid = spotid + 1


def __create_alltracks(model, df):
    # List of all tracks
    alltracks = ET.SubElement(model, "AllTracks")


#   for particle in df["particle"].unique():
#       df_track = df[df["particle"] == particle]
#       track_ids = list(df_track.index)
#       frames = np.array(df_track["slice"])
#       longest, total, duration = get_gaps(frames)
#       dict_track = {
#           "name": f"Track_{particle}",
#           "TRACK_ID": str(particle),
#           "NUMBER_SPOTS": str(len(frames)),
#           "NUMBER_GAPS": longest,
#           "LONGEST_GAP": total,
#           "NUMBER_SPLITS": "0",
#           "NUMBER_MERGES": "0",
#           "NUMBER_COMPLEX": "0",
#           "TRACK_DURATION": duration,
#           "TRACK_START": str(min(frames)),
#           "TRACK_STOP": str(max(frames)),
#           "TRACK_DISPLACEMENT": "0.01",
#           "TRACK_INDEX": str(particle),
#           "TRACK_X_LOCATION": str(df_track["x"].mean()),
#           "TRACK_Y_LOCATION": str(df_track["y"].mean()),
#           "TRACK_Z_LOCATION": "0.1",
#           "TRACK_MEAN_SPEED": "0.1",
#           "TRACK_MAX_SPEED": "0.1",
#           "TRACK_MIN_SPEED": "0.1",
#           "TRACK_MEDIAN_SPEED": "0.1",
#           "TRACK_STD_SPEED": "0.1",
#           "TRACK_MEAN_QUALITY": "0.1",
#           "TRACK_MAX_QUALITY": "0.1",
#           "TRACK_MIN_QUALITY": "0.1",
#           "TRACK_MEDIAN_QUALITY": "0.1",
#           "TRACK_STD_QUALITY": "0.1",
#       }
#       track = ET.SubElement(alltracks, "Track", **dict_track)

#       # Add all spots in the corresponding track
#       for row in df_track.iterrows():
#           dict_edge = {
#               "SPOT_SOURCE_ID": f"{row[0]:06}",
#               "SPOT_TARGET_ID": f"{track_ids[track_ids.index(row[0]) - 1]:06}",
#               "LINK_COST": "0.1",
#               "EDGE_TIME": "0.1",
#               "EDGE_X_LOCATION": str(row[1]["x"]),
#               "EDGE_Y_LOCATION": str(row[1]["y"]),
#               "EDGE_Z_LOCATION": "0.0",
#               "VELOCITY": "0.1",
#               "DISPLACEMENT": "0.1",
#           }
#           _ = ET.SubElement(track, "Edge", **dict_edge)


def __create_filteredtracks(model, df):
    # Tracks after TrackMate's filtering
    filteredtracks = ET.SubElement(model, "FilteredTracks")


#   for particle in df["particle"].unique():
#       _ = ET.SubElement(filteredtracks, "TrackID", TRACK_ID=str(particle))


def __create_settings(
    root,
    file_image,
    pixelwidth: str = "1.0",
    pixelheight: str = "1.0",
    voxeldepth: str = "1.0",
    timeinterval: str = "1.0",
):
    # Image metadata
    path, fname = os.path.split(file_image)
    image = skimage.io.imread(file_image)
    frames, width, height = image.shape
    imagedata = {
        "filename": fname,
        "folder": path,
        "width": str(width),
        "height": str(height),
        "nslices": "1",
        "nframes": str(frames),
        "pixelwidth": pixelwidth,
        "pixelheight": pixelheight,
        "voxeldepth": voxeldepth,
        "timeinterval": timeinterval,
    }
    basicsettings = {
        "xstart": "0",
        "xend": str(width - 1),
        "ystart": "0",
        "yend": str(height - 1),
        "zstart": "0",
        "zend": "0",
        "tstart": "0",
        "tend": str(frames - 1),
    }
    detectorsettings = {
        "DETECTOR_NAME": "LOG_DETECTOR",
        "TARGET_CHANNEL": "1",
        "RADIUS": "5.0",
        "THRESHOLD": "1000.0",
        "DO_MEDIAN_FILTERING": "false",
        "DO_SUBPIXEL_LOCALIZATION": "true",
    }
    initialspotfilter = {"feature": "QUALITY", "value": "0.0", "isabove": "true"}
    dict_trackersettings = {
        "TRACKER_NAME": "SPARSE_LAP_TRACKER",
        "CUTOFF_PERCENTILE": "0.9",
        "ALTERNATIVE_LINKING_COST_FACTOR": "1.05",
        "BLOCKING_VALUE": "Infinity",
    }
    dict_subtrackersettings = {
        "Linking": {"LINKING_MAX_DISTANCE": "0.8"},
        "GapClosing": {
            "ALLOW_GAP_CLOSING": "false",
            "GAP_CLOSING_MAX_DISTANCE": "0.5",
            "MAX_FRAME_GAP": "3",
        },
        "TrackSplitting": {
            "ALLOW_TRACK_SPLITTING": "false",
            "SPLITTING_MAX_DISTANCE": "15.0",
        },
        "TrackMerging": {
            "ALLOW_TRACK_MERGING": "false",
            "MERGING_MAX_DISTANCE": "15.0",
        },
    }
    dict_analyzercollection = {
        "SpotAnalyzers": [
            "MANUAL_SPOT_COLOR_ANALYZER",
            "Spot descriptive statistics",
            "Spot radius estimator",
            "Spot contrast and SNR",
        ],
        "EdgeAnalyzers": [
            "Edge target",
            "Edge mean location",
            "Edge velocity",
            "MANUAL_EDGE_COLOR_ANALYZER",
        ],
        "TrackAnalyzers": [
            "Branching analyzer",
            "Track duration",
            "Track index",
            "Track location",
            "Velocity",
            "TRACK_SPOT_QUALITY",
        ],
    }

    # General Settings
    settings = ET.SubElement(root, "Settings")
    _ = ET.SubElement(settings, "ImageData", **imagedata)
    _ = ET.SubElement(settings, "BasicSettings", **basicsettings)
    _ = ET.SubElement(settings, "DetectorSettings", **detectorsettings)
    _ = ET.SubElement(settings, "InitialSpotFilter", **initialspotfilter)
    _ = ET.SubElement(settings, "SpotFilterCollection")

    # Tracker settings
    trackersettings = ET.SubElement(settings, "TrackerSettings", **dict_trackersettings)
    for k, v in dict_subtrackersettings.items():
        subelement = ET.SubElement(trackersettings, k, **v)
        _ = ET.SubElement(subelement, "FeaturePenalties")

    # Filter settings
    _ = ET.SubElement(settings, "TrackFilterCollection")
    analyzercollection = ET.SubElement(settings, "AnalyzerCollection")
    for k, v in dict_analyzercollection.items():
        subanalyzer = ET.SubElement(analyzercollection, k)
        for lst in v:
            _ = ET.SubElement(subanalyzer, "Analyzer", key=lst)


def __create_guistate(root):
    # TrackMate's GUI settings
    guistate = ET.SubElement(root, "GUIState", state="InitialFiltering")
    for _ in range(4):
        _ = ET.SubElement(guistate, "View", key="HYPERSTACKDISPLAYER")


def __pretty_output(root, file_output):
    # Save file after fancy formatting to prettify
    with tempfile.TemporaryDirectory() as tempdirname:
        fname = os.path.join(tempdirname, "file.xml")
        tree = ET.ElementTree(root)
        tree.write(fname, encoding="UTF-8", xml_declaration=True)
        dom = xml.dom.minidom.parse(fname)
        pretty_xml = dom.toprettyxml()

        with open(file_output, "w") as f:
            f.write(pretty_xml)


def create_trackmate_xml(
    spots_df,
    file_image,
    file_output,
    spatialunits: str = "pixel",
    timeunits: str = "sec",
    pixelwidth: int = 1,
    pixelheight: int = 1,
    voxeldepth: int = 1,
    timeinterval: int = 1,
):
    # Check required track df columns
    df = spots_df
    df["x"] = df["x"] * pixelwidth
    df["y"] = df["y"] * pixelheight

    df.to_csv(file_output.replace("xml", "csv"))

    req_cols = ["x", "y", "slice"]
    if not all(req in df.columns for req in req_cols):
        raise ValueError(f"Not all required columns present! {req_cols} must exist.")

    # XML tree
    root = ET.Element("TrackMate", version="6.0.1")

    # Model
    model = __create_model(root, spatialunits=spatialunits, timeunits=timeunits)
    __create_allspots(model, df)
    __create_alltracks(model, df)
    __create_filteredtracks(model, df)

    # Settings
    __create_settings(
        root,
        file_image,
        pixelwidth=str(pixelwidth),
        pixelheight=str(pixelheight),
        voxeldepth=str(voxeldepth),
        timeinterval=str(timeinterval),
    )
    __create_guistate(root)

    # Save output
    __pretty_output(root, file_output)
