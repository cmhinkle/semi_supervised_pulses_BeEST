###############################################
# Original author: Calvin Hinkle
# Last modified date: 8/3/2026 
# Description: 
# Select N pulses from a day of data. Pulses have 
# been selected from pulseGrabberChan.py
# Can select across all channels and time chunks
# or from a specifc channel

# Select a number of pulses from a day uniformly distributed temportally
# If looking at multiple channels, samples uniformly across channels
###############################################

import numpy as np
import re
from tqdm.auto import tqdm
import glob
import argparse
from datetime import datetime
import os

########################################################################################################################################

SAVE_DIR = '/data_fast/chinkle/26_ml/selectedPulses/'

########################################################################################################################################

def _arg_config():
    parser = argparse.ArgumentParser(description='Select N pulses from a a day of data. Can choose a single channel or all channels')
    parser.add_argument(
        '--save_directory', 
        help='Directory to save N pulses to. The filename will automatically generated be based on filename, chans (if included), and date',
        default=SAVE_DIR,
        required=False
    )
    parser.add_argument(
        '--pulse_directory', 
        help='Directory to sample pulses from. Input data must follow from pulses saved using pulseGrabber.py', 
        required=True
    )
    parser.add_argument(
        '--n_samples', 
        type=int, 
        help='Numer of samples to save', 
        required=False, 
        default=5000
    )
    parser.add_argument(
        '--chans',
        type=int,
        nargs='+',
        help='Channel(s) to sample from. List with spaces. Default: all channels. Ex: 0 2 10',
        default=None
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )

    return parser

#######################################################################################################################################

# Search through all .npy files in a dir and save properally formatted
# files based on their channel and time chunk in a file_map
# Access based on file_map[time, chan]

def makeFileMap(path, channels=None):
    files = glob.glob(path.rstrip('/') + '/*.npy')
    
    file_map = {}

    pattern = re.compile(r"_(\d+)_(\d+)\.npy$")

    for f in files:
        m = pattern.search(f)
        if m:
            chunk = int(m.group(1))
            chan = int(m.group(2))

            if channels is None or chan in channels:
                file_map[(chunk, chan)] = f

    if not file_map:
        if channels is None:
            raise RuntimeError("No valid pulse files found.")
        else:
            raise RuntimeError(
                f"No files found for requested channel(s): {channels}"
            )

    return file_map

########################################################################################################################################
 
def getPulseDtype():
    prepulse_len = 2**7
    pulse_len = 2**11 - prepulse_len

    return np.dtype([
        ('chunk', np.int32),
        ('channel', np.int16),
        ('loc', np.int32),
        ('waveform', np.float32, (pulse_len + prepulse_len,)),
        ('ig_laser', np.bool_),
        ('multiplicity', np.int32),
        ('energy', np.float32)
    ])

########################################################################################################################################

def makeSaveName(args):

    date_str = datetime.now().strftime("%Y%m%d")
    m = re.search(r'out_pulses_(\d+)', args.pulse_directory)

    if m:
        day = int(m.group(1))
    else:
        raise RuntimeError(
            f'Could not determine day number from {args.pulse_directory}'
        )

    if args.chans is None:
        chan_str = "allChans"
    elif len(args.chans) == 1:
        chan_str = f"chan{args.chans[0]:02d}"
    else:
        chan_str = "chans_" + "-".join(
            f"{c:02d}" for c in sorted(args.chans)
        )

    save_loc = (
        args.save_directory.rstrip("/")
        + f"/day{day:02d}_{chan_str}_{args.n_samples}_{date_str}.npy"
    )

    count = 1
    base = save_loc[:-4]

    while os.path.exists(save_loc):
        save_loc = f"{base}_{count}.npy"
        count += 1

    return save_loc

########################################################################################################################################

def samplePulses(file_map, num_samples, dtype):

    selected = getPulses(file_map, num_samples)

    file_choices, counts = np.unique(
        selected,
        axis=0,
        return_counts=True
    )

    pulses = np.zeros(num_samples, dtype=dtype)

    i = 0

    for file_choice, num in tqdm(
        zip(file_choices, counts),
        total=len(file_choices)
    ):
        chunk_num, chan = file_choice
        chunk_data = np.load(file_map[(chunk_num, chan)])

        if len(chunk_data) == 0:
            continue

        for _ in range(num):

            pulseLoc = np.random.randint(len(chunk_data))

            pulses[i] = (
                chunk_num,
                chan,
                pulseLoc,
                chunk_data['waveform'][pulseLoc],
                chunk_data['ig_laser'][pulseLoc],
                chunk_data['multiplicity'][pulseLoc],
                chunk_data['energy'][pulseLoc]
            )

            i += 1

    if i != num_samples:
        print(
            f"Warning: only sampled {i}/{num_samples} pulses"
        )

    return pulses[:i]

########################################################################################################################################

# Select random time chunks and channels

def getPulses(file_map, num):
    if not file_map:
        raise RuntimeError("No valid chunks found.")

    keys = np.array(list(file_map.keys()))

    random_idx = np.random.randint(
        len(keys),
        size=num
    )

    return keys[random_idx]

########################################################################################################################################

def main():

    args = _arg_config().parse_args()

    os.makedirs(args.save_directory, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)

    file_map = makeFileMap(
        args.pulse_directory,
        args.chans
    )

    dtype = getPulseDtype()

    pulses = samplePulses(
        file_map,
        args.n_samples,
        dtype
    )

    save_loc = makeSaveName(args)

    np.save(save_loc, pulses)

    print(f"Saved at {save_loc}")

########################################################################################################################################

if __name__ == '__main__':
    main()
