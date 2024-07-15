import cv2
import plotly.express as px
import numpy as np
import pandas as pd
import hashlib

def cache(fn):
    def cached(*args, **kwargs):
        kwargs_keys = sorted(kwargs.keys())
        kwargs_vals = [kwargs[k] for k in kwargs_keys]
        key = (args, tuple(kwargs_vals))
        if key not in cached.cache:
            print(f'first time running with args {args} and kwargs {kwargs}')
            val = fn(*args, **kwargs)
            cached.cache[key] = val
        else:
            print(f'using cache for args {args} and kwargs {kwargs}')
            val = cached.cache[key]
        return cached.cache[key]
    cached.cache = {}
    return cached

@cache
def read_frames(path: str, max_frames: int = 5000, resize=(100,100)):
    stream = cv2.VideoCapture(path)
    frames = [cv2.resize(stream.read()[1].astype(float)/255, resize) for i in range(max_frames)]
    return np.stack(frames)


def get_key_frames(frames, threshold=0.2):
    diffs = frames[1:, :, :, :] - frames[:-1, :, :, :]
    delta = np.abs(diffs).mean(axis=(1,2,3))
    frames_indexes = np.where(delta>threshold)[0] + 1
    return frames[frames_indexes, :, :, :], frames_indexes


def hash_image(image, res=8):
    image = image.mean(axis=-1)
    image = cv2.resize(image, dsize=(res, res), interpolation=cv2.INTER_AREA)
    image = image > image.mean()
    return hashlib.md5(image).hexdigest()

paths = ['../static/The Office/season 4/The Office (US) (2005) - S04E01-E02 - Fun Run (1080p BluRay x265 Silence).mkv',
         '../static/The Office/season 4/The Office (US) (2005) - S04E03-E04 - Dunder Mifflin Infinity (1080p BluRay x265 Silence).mkv']
frames = read_frames(paths[0])
frames2 = read_frames(paths[0])
pass
