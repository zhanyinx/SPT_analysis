# @ String xml
# @ String basedir

from java.io import File
import sys
import csv
import os

from fiji.plugin.trackmate.io import TmXmlReader
from fiji.plugin.trackmate.io import TmXmlWriter
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate.providers import DetectorProvider
from fiji.plugin.trackmate.providers import TrackerProvider
from fiji.plugin.trackmate.providers import SpotAnalyzerProvider
from fiji.plugin.trackmate.providers import EdgeAnalyzerProvider
from fiji.plugin.trackmate.providers import TrackAnalyzerProvider


def process(trackmate):
    """
    Execute the full process BUT for the detection step.
    """
    # Check settings.
    ok = trackmate.checkInput()
    # Compute spot features.
    print("Computing spot features")
    ok = ok and trackmate.computeSpotFeatures(True)
    # Track spots.
    print("Tracking")
    ok = ok and trackmate.execTracking()
    # Compute track features.
    print("Computing track features")
    ok = ok and trackmate.computeTrackFeatures(True)
    # Filter tracks.
    print("Filtering tracks")
    ok = ok and trackmate.execTrackFiltering(True)
    # Compute edge features.
    print("Computing link features")
    ok = ok and trackmate.computeEdgeFeatures(True)

    return ok


# ----------------
# Setup variables
# ----------------

file = File(xml)

# -------------------
# Instantiate reader
# -------------------

reader = TmXmlReader(file)
if not reader.isReadingOk():
    sys.exit(reader.getErrorMessage())
# -----------------
# Get a full model
# -----------------

# This will return a fully working model, with everything
# stored in the file. Missing fields (e.g. tracks) will be
# null or None in python
model = reader.getModel()
# model is a fiji.plugin.trackmate.Model


# ---------------------------------------
# Building a settings object from a file
# ---------------------------------------

# Reading the Settings object is actually currently complicated. The
# reader wants to initialize properly everything you saved in the file,
# including the spot, edge, track analyzers, the filters, the detector,
# the tracker, etc...
# It can do that, but you must provide the reader with providers, that
# are able to instantiate the correct TrackMate Java classes from
# the XML data.

# We start by creating an empty settings object
settings = Settings()

# Then we create all the providers, and point them to the target model:
detectorProvider = DetectorProvider()
trackerProvider = TrackerProvider()
spotAnalyzerProvider = SpotAnalyzerProvider()
edgeAnalyzerProvider = EdgeAnalyzerProvider()
trackAnalyzerProvider = TrackAnalyzerProvider()

# Ouf! now we can flesh out our settings object:
reader.readSettings(
    settings,
    detectorProvider,
    trackerProvider,
    spotAnalyzerProvider,
    edgeAnalyzerProvider,
    trackAnalyzerProvider,
)

# overwrite tracking parameters
settings.trackerSettings["LINKING_MAX_DISTANCE"] = 0.8
settings.trackerSettings["ALLOW_TRACK_SPLITTING"] = False
settings.trackerSettings["ALLOW_TRACK_MERGING"] = False
settings.trackerSettings["ALLOW_GAP_CLOSING"] = True
settings.trackerSettings["GAP_CLOSING_MAX_DISTANCE"] = 0.8
settings.trackerSettings["MAX_FRAME_GAP"] = 1



trackmate = TrackMate(model, settings)
ok = process(trackmate)

if not ok:
    sys.exit(str(trackmate.getErrorMessage()))
    
imp = settings.imp
fname = str(imp.title)

# Export model to xml
outFile_name = fname + ".tracks.xml" 
outFile = File(basedir, outFile_name)

writer = TmXmlWriter(outFile) #a File path object
writer.appendModel( trackmate.getModel() ) #trackmate instantiate like this before trackmate = TrackMate(model, settings)
writer.appendSettings( trackmate.getSettings() )
writer.writeToFile()

spots = [['track', 'x', 'y', 'z', 'frame', 'cell']]


# Export as csv
# Get spot coordinate and id
for id in model.getTrackModel().trackIDs(True):
   
       
    track = model.getTrackModel().trackSpots(id)
    for spot in track:
        sid = spot.ID()
        # Fetch spot features directly from spot. 
        x=spot.getFeature('POSITION_X')
        y=spot.getFeature('POSITION_Y')
        z=spot.getFeature('POSITION_Z')
        t=spot.getFeature('POSITION_T')
       
    	spots.append([id, x, y, 
                      z, t, -1 ])
#write output
with open(basedir + "/" + fname + ".tracks.csv", "wb") as f:
    wr = csv.writer(f)
    for row in spots:
        wr.writerow(row)


