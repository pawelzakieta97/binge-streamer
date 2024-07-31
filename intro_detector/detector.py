import argparse
import os.path
import pickle

import cv2
import plotly.express as px
import numpy as np
import pandas as pd
import hashlib
from argparse import ArgumentParser

def cache(persistent=True):
    CACHE_DIR = 'cache'

    def get_key(args, kwargs):
        kwargs_keys = sorted(kwargs.keys())
        kwargs_vals = [kwargs[k] for k in kwargs_keys]
        return (args, tuple(kwargs_vals))

    def get_cache_file(args, kwargs):
        os.makedirs(CACHE_DIR, exist_ok=True)
        key = get_key(args, kwargs)
        return os.path.join(CACHE_DIR, hashlib.md5(str(key).encode()).hexdigest() + '.pickle')

    def get_cached(cache_dict, args, kwargs):
        key = get_key(args, kwargs)
        if key in cache_dict:
            return cache_dict[key]
        if persistent:
            cache_file = get_cache_file(args, kwargs)
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        return None

    def save_cache(cache_dict, args, kwargs, val):
        key = get_key(args, kwargs)
        cache_dict[key] = val
        if persistent:
            with open(get_cache_file(args, kwargs), 'wb') as f:
                pickle.dump(val, f)

    def decorator(fn):
        def cached(*args, **kwargs):
            val = get_cached(cached.cache, args, kwargs)
            if val is None:
                print(f'first time running with args {args} and kwargs {kwargs}')
                val = fn(*args, **kwargs)
                save_cache(cached.cache, args, kwargs, val)
            else:
                print(f'using cache for args {args} and kwargs {kwargs}')
            return val
        cached.cache = {}
        return cached
    return decorator


@cache(persistent=True)
def read_frames(path: str, max_frames: int = 5000, resize=(100,100)):
    stream = cv2.VideoCapture(path, apiPreference=cv2.CAP_FFMPEG)
    fps = stream.get(cv2.CAP_PROP_FPS)
    frames = [cv2.resize(stream.read()[1].astype(float)/255, resize) for i in range(max_frames)]
    return np.stack(frames), fps


def get_key_frames(frames, threshold=0.15):
    diffs = frames[1:, :, :, :] - frames[:-1, :, :, :]
    delta = np.abs(diffs).mean(axis=(1,2,3))
    # todo: smarter way to determine keyframes (adaptive threshold)
    delta2 = delta[1:] - delta[:-1]
    frames_indexes = np.where(delta>threshold)[0] + 1
    return frames[frames_indexes, :, :, :], frames_indexes


def hash_image(image, res=8):
    image = image.mean(axis=-1)
    image = cv2.resize(image, dsize=(res, res), interpolation=cv2.INTER_AREA)
    image = image > image.mean()
    return hashlib.md5(image).hexdigest()


def dilate(a: np.array, size: int = 1):
    kernel_size = 2 * size + 1
    ii = np.arange(a.shape[0]) + (np.arange(kernel_size) - size)[:, None]
    ii[ii<0] = 0
    ii[ii>=a.shape[0]] = a.shape[0] - 1
    return a[ii].any(axis=0)


def erode(a: np.array, size: int = 1):
    kernel_size = 2 * size + 1
    ii = np.arange(a.shape[0]) + (np.arange(kernel_size) - size)[:, None]
    ii[ii<0] = 0
    ii[ii>=a.shape[0]] = a.shape[0] - 1
    return a[ii].all(axis=0)

def save_video(data: list[np.array], path: str):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
    for frame in data:
        out.write(frame)
    out.release()


class IntroDetector:
    def __init__(self, hash_options: dict = None):
        self.frame_hashes = {}
        if hash_options is None:
            hash_options = {}
        self.hash_options = hash_options
        self.keyframes = {}
        self.result = None

    def update(self, frames, name: str, fps: float = 1.0):
        keyframes, frame_indexes = get_key_frames(frames)
        self.keyframes[name] = []
        for kf_index, (kf, ts) in enumerate(zip(keyframes, frame_indexes)):
            h = hash_image(kf, **self.hash_options)
            if h not in self.frame_hashes:
                self.frame_hashes[h] = []
            self.frame_hashes[h].append((name, kf_index))
            self.keyframes[name].append({'ts': ts/fps, 'kf': kf})

    def detect(self, threshold=0.5, morph=2):
        shot_repeats_for_intro = max(2, int(len(self.keyframes) * threshold))
        for frame_hash, occurrences in self.frame_hashes.items():
            for name, kf_index in occurrences:
                self.keyframes[name][kf_index]['count'] = len(occurrences)
        result = {}
        for name, keyframes in self.keyframes.items():
            mask = np.array([kf_dict['count'] >= shot_repeats_for_intro for kf_dict in keyframes])
            if morph > 0:
                mask = erode(dilate(mask, morph), morph)

            data = self.keyframes[name]
            for d, m in zip(data, mask):
                d['intro'] = m
            # TODO: dont use pandas, for some reason slow af
            result[name] = pd.DataFrame(data).drop(columns=['kf'])
        self.result = result
        return result

    def save(self, save_intros=False):
        for file_path, df in self.result.items():
            intro = df.intro.values
            intro[-1] = 0
            if not intro.any():
                continue
            intro_begins = np.where(intro[1:] & ~intro[:-1])[0] + 1
            intro_ends = np.where(~intro[1:] & intro[:-1])[0] + 1
            intro_lengths = intro_ends - intro_begins
            max_intro_idx = np.argmax(intro_lengths)
            intro_start = intro_begins[max_intro_idx]
            intro_end = intro_ends[max_intro_idx]
            intro_start_frame = float(df.ts[intro_start])
            intro_end_frame = float(df.ts[intro_end])
            with open(os.path.splitext(file_path)[0]+'_intro.txt', 'w+') as f:
                f.writelines([str(intro_start_frame)+'\n', str(intro_end_frame)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog='detector',
                        description='detects intros in selected folder')
    parser.add_argument('dir', default='.')  # positional argument
    args = parser.parse_args()
    print(args.dir)
    video_dir = args.dir
    paths = [os.path.join(video_dir, video_file) for video_file in os.listdir(video_dir) if video_file.endswith('.mkv')]# and 'Stutter' in video_file]
    frames, fpses = zip(*[read_frames(path) for path in paths])


    detector = IntroDetector()
    for i, (fr, fps) in enumerate(zip(frames, fpses)):
        print(i)
        detector.update(fr, paths[i], fps)

    res = detector.detect()
    detector.save(save_intros=True)
