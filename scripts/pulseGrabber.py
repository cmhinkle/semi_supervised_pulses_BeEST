###############################################
# Adapted by: Calvin Hinkle 
# Original author: Drew Marino 
# Last modified date: 8/3/2026 
# Update for channel 3, day 16: 7/14/2026
# Description: 
# Select pulses for each each time chunk (~10 min) 
# for each channel and save to .npy
# Can choose to save all channels or specify 
# Saves:
# .npy  for each chan/chunk combo at save_directory
###############################################

from nptdms import TdmsFile
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import glob
import os
import argparse

########################################################################################################################################

# I am looking at day 16, channel 3
# /beest_data/20221019-20221227_FirstCollectionRun/magcycle-16-221106/rawContinuousDAQ-2022-11-07

H5_BASE_PATH = r'/beest_data/finalProcessingBlinded-20250519/magcycle-16-221106/processed/'

TDMS_BASE_PATH = r'/beest_data/20221019-20221227_FirstCollectionRun/magcycle-16-221106/rawContinuousDAQ-2022-11-07/'

########################################################################################################################################

# Just need tdms_dir, h5_dir, save_dir, threads, chans
def _arg_config():
    parser = argparse.ArgumentParser(
        description='Generate template pulses for each channel with configurable energy range and time aggregation'
    )
    parser.add_argument(
        '--tdms_directory', 
        help='Directory where raw TDMS files are stored', 
        default=TDMS_BASE_PATH, 
        required=False
    )
    parser.add_argument(
        '--h5_directory', 
        help='Directory where associated h5 files are stored', 
        default=H5_BASE_PATH, 
        required=False
    )
    parser.add_argument(
        '--save_directory', 
        help='Directory where template pulses will be stored',       
        default=None,
        required=False
    )    
    parser.add_argument(
        '--threads', 
        help='Max threads to launch. Max 48 cores on CRONOS', 
        default=20, 
        required=False, 
        type=int
    )
    parser.add_argument(
        '--chans', 
        nargs='*', 
        type=int, 
        default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        help='Channels to save pulses from. Seperate using spaces. Ex. --chans 0 1 2', 
        required=False
    )

    return parser
    
########################################################################################################################################

def make_save_directory(args):
    # Extract day number from path like:
    # /data_fast/SharedFiles/chew5/magcycle-16-221106/processed/
    day = None

    for part in args.h5_directory.rstrip('/').split('/'):
        if part.startswith('magcycle-'):
            day = part.split('-')[1]
            break

    if day is None:
        day = 'unknown'

    # Generate directory name based on channels
    all_chans = list(range(16))

    if sorted(args.chans) == all_chans:
        chan_string = 'all'
    else:
        chan_string = 'chans_' + '_'.join(str(c) for c in sorted(args.chans))

    return f'/data_fast/chinkle/26_ml/allPulses/out_pulses_{day}_{chan_string}/'

########################################################################################################################################

# Adapted function to load h5 and tdms file and save the pulses
def getPulses(sortedNum, h5Path, prepulse, pulselen, args):
    #print(f'starting {sortedNum}')
    # From an h5 in the form: 
    # /data_fast/SharedFiles/chew5/magcycle-33-221128/processed/chewed_20221129-163127.671915_Sig_A.h5
    # Saving files in the form, with padding for channel and sortedNum
    # 20221129-163127.671915_{A/B}_{sortedNum}_{channel}.npy
    chans = args.chans
    save_loc_base = args.save_directory
    save_loc_base += h5Path.split('/')[-1].split('_')[-3] + '_' 
    save_loc_base += h5Path[-4] + '_' #A or B
    save_loc_base += f'{str(sortedNum).zfill(3)}'
    buf = (ord(h5Path[-4])-65)*8 # if 'B' add 8 to chan, 'A' add 0 to chan
    save_locs = [save_loc_base + f'_{str(channel + buf).zfill(2)}.npy' for channel in range(8)]
    for save_loc in save_locs:
        if os.path.exists(save_loc):
            print(f'file exists: {save_loc} ')
            return None  # Don't recreate files that already exist. 
                         # This is maybe not desirable but allows to rerun if it crashes without redoing everything
    
    current_path = ''    
    try:
        current_path = h5Path
        tdmsPath = args.tdms_directory + ((h5Path.split('/')[-1])[len('chewed_metadata_'):-3]) + '.tdms'
        with TdmsFile.open(tdmsPath) as TDMS:
            with pd.HDFStore(h5Path, mode='r') as hdf:
                for _channel in hdf.keys():
                    # Using HDF trigger locations to extract pulses from TDMS files
                    if 'metadata' not in _channel:
                        _chan_num = int(_channel[13:]) # takes # from /data_channel#
                        if 'A' in h5Path[-5:]:
                            chn = TDMS.groups()[0]['Dev1/ai' + str(_channel[13:])] # takes # from /data_channel#
                        if 'B' in h5Path[-5:]:
                            chn = TDMS.groups()[0]['PXI1Slot9/ai' + str(int(_channel[13:]) - 8)]

                        if _chan_num in chans:
                            #tdms_channel = chn[:] # much faster but huge RAM usage for many workers
                            #df = hdf[_channel]
                            #print(f'chan: {_channel}')
                            #print(f'save at: {save_locs[int(_channel[13:])%8]}')
                    
                            # Laser has an inherent offset that I need to correct for per Inwook
                            #p0 = hdf.select(key='meta'+_channel[1:]).calibration_parameters[-1][-1]
                            #print(f'p0 good: {p0}')
    
                            trigs = hdf.select(key=_channel).trigger_position_corrected.to_numpy()
                            ig_lasers = hdf.select(key=_channel).ig_laser.to_numpy()
                            mults = hdf.select(key=_channel).multiplicity.to_numpy()
                            #es = hdf.select(key=_channel).calibrated_energy.to_numpy()
                            es = hdf.select(key=_channel).drift_corrected_energy_substr_corr_rescaled.to_numpy()
                            
                            if len(trigs) == 0:
                                print(f'No pulses found in chan {_channel}')
                                continue
                                
                            newType = np.dtype([
                                ('waveform', np.float32, (prepulse+pulselen,)),
                                ('ig_laser', np.bool_),
                                ('multiplicity', np.int32),
                                ('energy', np.float32)
                            ])
            
                            _to_return = np.empty(len(trigs), dtype=newType) # cols: waveform, ig_laser, multiplicity, calibrated_energy
    
                            # not optimal for speed, but better for RAM
                            for i, trig in enumerate(trigs):
                                start = int(trig - prepulse)
                                stop  = int(trig + pulselen)
                            
                                _to_return['waveform'][i] = chn[start:stop]   
                            
                            _to_return['ig_laser'] = ig_lasers
                            _to_return['multiplicity'] = mults
                            _to_return['energy'] = es
                            
                            #print(_to_return.shape)
                            np.save(save_locs[int(_channel[13:])%8], _to_return) # save each chan to a file        
        
    except ValueError: # This line is reached if the end of a tdms file is corrupt. I believe Connor had a solution to this but I didn't implement it.
        print(f"ValueError occurred for time {sortedNum}: {(current_path.split('/'))[-1]}")
        pass

########################################################################################################################################

def main():
    args = _arg_config().parse_args()

    if args.save_directory is None:
        args.save_directory = make_save_directory(args)
    
    h5PathsA = sorted(glob.glob(args.h5_directory + '*A.h5'))
    h5PathsB = sorted(glob.glob(args.h5_directory + '*B.h5'))

    # Total window should be some 2^n ant these to add up to a power of 2 for DWT
    # For DWT, rising edge should be around a power of 2
    prepulse_len = int(2**7)
    pulse_len = int(2**11 - prepulse_len)

    if not os.path.isdir(args.save_directory):
        os.makedirs(args.save_directory)

    print(f'Saving pulses to: {args.save_directory}')

    tasks = []

    for _i, (h5A, h5B) in enumerate(zip(h5PathsA, h5PathsB)):
        tasks.append((_i, h5A))
        tasks.append((_i, h5B))
    
    res = Parallel(
        n_jobs=args.threads, 
        backend='multiprocessing',
        verbose=10
    )(delayed(getPulses)(
            sortedNum,
            h5path,
            prepulse_len,
            pulse_len,
            args
        )
        for sortedNum, h5path in tasks
    )

########################################################################################################################################

if __name__ == '__main__':
    main()
