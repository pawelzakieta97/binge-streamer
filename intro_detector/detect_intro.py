import cv2
import plotly.express as px
import numpy as np
end_frame = 24 * 100
stream = cv2.VideoCapture('../static/The Office/season 4/The Office (US) (2005) - S04E01-E02 - Fun Run (1080p BluRay x265 Silence).mkv')
frame = 0
differences = []
ds = []
prev = None
threshold = 0.2
key_frames = []
while True:
    success, image = stream.read()
    downscaled = cv2.resize(image, (100,100)).astype(float) / 255
    if prev is not None:
        diff = np.abs(downscaled - prev)
        differences.append(diff)
        ds.append(diff.mean())
        if ds[-1] > threshold:
            key_frames.append(image)
    prev = downscaled
    frame += 1
    if frame > end_frame:
        break
    if not success:
        break

px.line(ds).show()
