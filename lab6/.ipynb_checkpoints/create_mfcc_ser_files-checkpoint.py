import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import scipy

import sys

import librosa
import librosa.display
import torch
from sklearn.preprocessing import OneHotEncoder

from IPython.display import Audio
import noisereduce as nr

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pickle as pkl
import multiprocessing as mp
from numba import jit
import time
import tqdm
import os

import concurrent.futures


def load_data(path, columns_to_drop=['hash_id', 'source_id']):
    df = pd.read_json(path, lines=True).drop(columns=columns_to_drop)
    df['audio_path'] = df['audio_path'].apply(lambda x: x[5:])
    df = df[df['annotator_emo'] != 'other']
    df = df[df['duration'] <= 5.0]
    return df


def create_mfcc_from_path(path, sr=15000):
    if not os.path.exists(path):
        return None
    data, sr = librosa.load(path, sr=sr)
    data = nr.reduce_noise(data, sr=sr)
    total_length = sr*5
    xt, index = librosa.effects.trim(data, top_db=33)
    xt = np.pad(xt[:total_length], (0, total_length-len(xt)), 'constant')
    mfcc = librosa.feature.mfcc(y=xt, sr=sr, n_mfcc=13, hop_length=512)
    return mfcc


def create_df_ready_for_serial2(args):
    df, df_type = args
    path_type = {
    'crowd_train': r'D:\data\dusha\crowd\crowd_train\wavs',
    'crowd_test': r'D:\data\dusha\crowd\crowd_test\wavs',
    'podcast_train': r'D:\data\dusha\podcast\podcast_train\wavs',
    'podcast_test': r'D:\data\dusha\podcast\podcast_test\wavs'
    }
    start_of_path = path_type[df_type]
    df = df.drop(columns=['golden_emo', 'annotator_id', 'speaker_emo'])
    df['audio_path'] = start_of_path + '\\' + df['audio_path']
    t = time.time()
    with mp.Pool(mp.cpu_count() - 1) as pool:
        res = list(tqdm.tqdm(pool.imap(create_mfcc_from_path, df['audio_path'])))
    df['mfcc'] = res
    print(f'time spent on mfcc = {round(time.time() - t, 3)}')
    return df


def serialize(args):
    df, df_type = args
    path = r'D:\data\serialized' + '\\' + df_type + '.ser'
    with open(path, 'wb') as f:
        pkl.dump(df, f)


def main2():

    try:
        crowd_train = load_data(r'D:\data\dusha\crowd\crowd_train\raw_crowd_train.jsonl')
        print('working of 1st')
        crowd_train = create_df_ready_for_serial2((crowd_train, 'crowd_train'))
        serialize((crowd_train, 'crowd_train'))
        del crowd_train
    except Exception as e:
        print(e)
    try:
        crowd_test = load_data(r'D:\data\dusha\crowd\crowd_test\raw_crowd_test.jsonl')
        print('working of 2nd')
        crowd_test = create_df_ready_for_serial2((crowd_test, 'crowd_test'))
        serialize((crowd_test, 'crowd_test'))
        del crowd_test
    except Exception as e:
        print(e)
    try:
        podcast_train = load_data(r'D:\data\dusha\podcast\podcast_train\raw_podcast_train.jsonl')
        print('working of 3rd')
        podcast_train = create_df_ready_for_serial2((podcast_train, 'podcast_train'))
        serialize((podcast_train, 'podcast_train'))
        del podcast_train
    except Exception as e:
        print(e)
    try:
        podcast_test = load_data(r'D:\data\dusha\podcast\podcast_test\raw_podcast_test.jsonl')
        print('working of 4th')
        podcast_test = create_df_ready_for_serial2((podcast_test, 'podcast_test'))
        serialize((podcast_test, 'podcast_test'))
        del podcast_test
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main2()