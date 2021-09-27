#@ String xml
#@ String basedir
#@ String maskfile
#@ String labeledFrames

from java.io import File
import sys
import os
import math
import csv

from ij import IJ
from fiji.plugin.trackmate.io import TmXmlReader
from fiji.plugin.trackmate.io import TmXmlWriter
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate.providers import DetectorProvider
from fiji.plugin.trackmate.providers import TrackerProvider
from fiji.plugin.trackmate.providers import SpotAnalyzerProvider
from fiji.plugin.trackmate.providers import EdgeAnalyzerProvider
from fiji.plugin.trackmate.providers import TrackAnalyzerProvider


def getClosestFrame(current, frameList):
    diffs = [math.fabs(current - int(l)) for l in frameList]
    return diffs.index(min(diffs))


def getCellID(ip, x, y, cal):
    return ip.getPixel(int(x / cal.pixelWidth), int(y / cal.pixelHeight))


def most_frequent(List):
    return max(set(List), key=List.count)


file = File(xml)  # read xml

reader = TmXmlReader(file)
if not reader.isReadingOk():
    sys.exit(reader.getErrorMessage())

model = reader.getModel()  # get the model
settings = Settings()  # initiate settings

# Feed settings
detectorProvider = DetectorProvider()
trackerProvider = TrackerProvider()
spotAnalyzerProvider = SpotAnalyzerProvider()
edgeAnalyzerProvider = EdgeAnalyzerProvider()
trackAnalyzerProvider = TrackAnalyzerProvider()

reader.readSettings(
    settings,
    detectorProvider,
    trackerProvider,
    spotAnalyzerProvider,
    edgeAnalyzerProvider,
    trackAnalyzerProvider,
)


## Assign cell id ##########

spots = model.getSpots()  # get spots

mask = IJ.openImage(maskfile)  # open mask file and get calibration
cal = settings.imp.getCalibration()
# frameList = list(range(mask.getNSlices()))
frameList = labeledFrames.split(",")  # get annotated frame list

# for each spot, look up cellID in closest frame
# put feature CellID
for spot in spots.iterable(False):
    x = spot.getDoublePosition(0)
    y = spot.getDoublePosition(1)
    frame = int(spot.getFeature("FRAME"))
    targetSlice = getClosestFrame(frame, frameList)
    ip = mask.getImageStack().getProcessor(targetSlice + 1)
    spot.putFeature("MANUAL_INTEGER_SPOT_FEATURE", getCellID(ip, x, y, cal))

trackModel = model.getTrackModel()
trackIDs = trackModel.trackIDs(False)
featureModel = model.getFeatureModel()

for trackID in trackIDs:
    trackSpots = trackModel.trackSpots(trackID)
    cellIDs = []
    for spot in trackSpots:
        cellIDs.append(int(spot.getFeature("MANUAL_INTEGER_SPOT_FEATURE")))

    dominantID = most_frequent(cellIDs)

    featureModel.putTrackFeature(trackID, "MANUAL_INTEGER_TRACK_FEATURE", dominantID)

trackmate = TrackMate(model, settings)

# Export model to xml
imp = settings.imp
fname = str(imp.title)

outFile_name = fname + ".tracks.with_cellIDs.xml"
outFile = File(basedir, outFile_name)
writer = TmXmlWriter(outFile)  # a File path object
writer.appendModel(
    trackmate.getModel()
)  # trackmate instantiate like this before trackmate = TrackMate(model, settings)
writer.appendSettings(trackmate.getSettings())
writer.writeToFile()


# Export in csv
spots = [['Label', 'ID', 'track', 'QUALITY', 'x', 'y',
       'z', 'POSITION_T', 'frame', 'RADIUS', 'VISIBILITY',
       'MANUAL_INTEGER_SPOT_FEATURE', 'MANUAL_DOUBLE_SPOT_FEATURE',
       'HAS_MAX_QUALITY_IN_FRAME', 'MANUAL_COLOR', 'MEAN_INTENSITY',
       'MEDIAN_INTENSITY', 'MIN_INTENSITY', 'MAX_INTENSITY', 'TOTAL_INTENSITY',
       'STANDARD_DEVIATION', 'ESTIMATED_DIAMETER', 'CONTRAST', 'SNR','cell']]


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
        f=spot.getFeature('FRAME')
        q=spot.getFeature('QUALITY')
        intensity=spot.getFeature('TOTAL_INTENSITY')
        mintensity=spot.getFeature('MEDIAN_INTENSITY')
        snr=spot.getFeature('SNR') 
        cellid=featureModel.getTrackFeature(id, 'MANUAL_INTEGER_TRACK_FEATURE')
    	spots.append(["ID" + str(sid) , sid, id, q, x, y, 
                      z, t, f, -1 , -1, 
                      -1, -1,
                      -1, -1, -1,
                      mintensity, -1, -1, intensity,
                      -1, -1, -1, snr,cellid])
#write output
with open(basedir + "/" + fname + ".tracks.with_cellIDs_uncorrected.csv", "wb") as f:
    wr = csv.writer(f)
    for row in spots:
        wr.writerow(row)


