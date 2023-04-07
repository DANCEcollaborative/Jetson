import os
import shutil
import time
import argparse

import numpy as np
from datetime import datetime

# (V) audio
import usb.core
import usb.util
import time

import pyaudio
import wave
import numpy as np
from usb_4_mic_array.get_index import get_respeaker_index
from usb_4_mic_array.tuning import Tuning

import sys
required_import_paths = ["~/", "/usr/local/lib/python3.6/pyrealsense2", "~/.local/lib/python3.6/site-packages"]
sys.path = sys.path + required_import_paths


dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
Mic_tuning = None
if dev:
    Mic_tuning = Tuning(dev)

def main(args):

    RESPEAKER_RATE = 16000
    RESPEAKER_CHANNELS = args.num_channels # change base on firmwares, 1_channel_firmware.bin as 1 or 6_channels_firmware.bin as 6
    RESPEAKER_WIDTH = 2
    # run getDeviceInfo.py to get index
    RESPEAKER_INDEX = get_respeaker_index()  # refer to input device id
    CHUNK = 1024
    WAVE_OUTPUT_FILE_PREFIX = "output_channel_"

    start_date_time = datetime.now().strftime('%m-%d-%y_%H:%M:%S')
    audio_dir = os.path.join(args.outdir, "audio", start_date_time)
    os.makedirs(audio_dir)

    # configure audio pipeline
    p = pyaudio.PyAudio()

    stream = p.open(
                rate=RESPEAKER_RATE,
                format=p.get_format_from_width(RESPEAKER_WIDTH),
                channels=RESPEAKER_CHANNELS,
                input=True,
                input_device_index=RESPEAKER_INDEX,)

    audio_output_files = []
    dirs_file = open(os.path.join(audio_dir, "doa.txt"), "a+")
    for i in range(RESPEAKER_CHANNELS):
        wf = wave.open(os.path.join(audio_dir, WAVE_OUTPUT_FILE_PREFIX+f"{i}.wav"), 'ab')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
        wf.setframerate(RESPEAKER_RATE)
        audio_output_files.append(wf)
        # wf.writeframes(b''.join(audio_frames[i]))
        # wf.close()

    # configure camera pipeline

    start = time.time()
    frames_saved = 0
    print('Recording started. Press ("Ctrl+C" ONLY TO STOP RECORDING.)')
    try:
        while True:
            data = stream.read(CHUNK)
            # extract channel 0 data from 6 channels, if you want to extract channel 1, please change to [1::6]
            for i in range(RESPEAKER_CHANNELS):
                channel_audio = np.fromstring(data, dtype=np.int16)[i::RESPEAKER_CHANNELS].tostring()
                audio_output_files[i].writeframes(channel_audio)
            if Mic_tuning is not None:
                dirs_file.write(f'{Mic_tuning.direction}\n')

            frames_saved += 1
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        print('Stopping Recording')
        stream.stop_stream()
        stream.close()
        p.terminate()
        end = time.time()
        for i in range(RESPEAKER_CHANNELS):
            audio_output_files[i].close()

        # np.save(os.path.join(audio_dir, "doa.npy"), np.array(dirs))
        dirs_file.close()
        print(f'Closed all open doa file and audio file pointers. Number of chunks recorded = {frames_saved}. Sample rate = {RESPEAKER_RATE}. Num channels saved = {RESPEAKER_CHANNELS}')
        print('Audio duration saved: ', (end-start)) # report frames per second

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='./output/', help='Name of output directory, must be created before running')
    parser.add_argument('--num_channels', default=6, type=int, help='Number of channels being recorded by microphone')
    parser.add_argument('--outfile', default='video', help='Name of video folder for output')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize frames while recording, cannot do in headless mode')
    args = parser.parse_args()
    main(args)