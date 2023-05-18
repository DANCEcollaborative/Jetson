import os
import shutil
import time
import argparse

import numpy as np

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

import zmq, datetime, time, json, msgpack
from datetime import timedelta
from zmq_utils import *

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
Mic_tuning = None
if dev:
    Mic_tuning = Tuning(dev)

def main(args):

    send_IPs_to_PSI()

    RESPEAKER_RATE = 16000 # sample rate of the audio
    RESPEAKER_CHANNELS = args.num_channels # change base on firmware, 1_channel_firmware.bin as 1 or 6_channels_firmware.bin as 6
    RESPEAKER_WIDTH = 2 # bytes, 1 byte = 8 bits, how many bytes to use to represent audio level for each sample
    # run getDeviceInfo.py to get index
    RESPEAKER_INDEX = get_respeaker_index()  # refer to input device id
    CHUNK = 1024 # number of samples recorded at each poll of micrphone
    WAVE_OUTPUT_FILE_PREFIX = "output_channel_"

    socket = create_socket(ip_address='tcp://*:40001') # channel to send audio
    socket2 = create_socket(ip_address='tcp://*:40002') # channel to send DOA
    socket3 = create_socket(ip_address='tcp://*:40003') # channel to send VAD

    start_date_time = datetime.datetime.now().strftime('%m-%d-%y_%H:%M:%S')
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
    dirs_file = open(os.path.join(audio_dir, "doa.txt"), "w+") # txt file t store every polled DOA value

    # to configure and create the wave file objects to write to for each of the 6 channels,  not recording audio at this stage, 
    for i in range(RESPEAKER_CHANNELS):
        wf = wave.open(os.path.join(audio_dir, WAVE_OUTPUT_FILE_PREFIX+f"{i}.wav"), 'wb')
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
            data = stream.read(CHUNK, exception_on_overflow=False) # if TRUE, the audio recording could stop automatically if the audio data exceeds system buffer capacity

            # extract channel 0 data from 6 channels, if you want to extract channel 1, please change to [1::6]
            # data = [0, 1, 2, 3, 4, 5, 0a, 1a, 2a, 3a, 4a, 5a, 0b, 1b, 2b, 3b, 4b, 5b, ....] where 0, 0a, 0b, 0c,.. represent subsequent audio samples from the 0th channel, and similarly for each of the 6 channels

            originatingTime = None
            for i in range(RESPEAKER_CHANNELS):
                channel_audio = np.fromstring(data, dtype=np.int16)[i::RESPEAKER_CHANNELS].tostring() 
                # np tostring() is an alias for np tobytes(); it returns bytes as opposed to str as implied by the function name
                audio_output_files[i].writeframes(channel_audio)
                if i==0:
                    # only sending audio from channel 0 to PSI
                    originatingTime = send_payload(socket, "temp", channel_audio)
                    print(f"Channel 0 audio sent at {originatingTime}", len(channel_audio))
                    
            if Mic_tuning is not None:
                dirn = Mic_tuning.direction
                vad = Mic_tuning.is_speech
                # explicitly setting the originatingTime for the payloads so that PSI receives and perceives the audio, DOA and VAD to be recorded at the same time. 
                originatingTime2 = send_payload(socket2, "temp2", dirn, originatingTime)
                originatingTime3 = send_payload(socket3, "temp3", vad, originatingTime2)
                print(f"DOA {dirn} - VAD {vad} | sent at {originatingTime2}", originatingTime==originatingTime2)
                dirs_file.write(f'DIR - {dir}; VAD - {vad};\n')
                dirs_file.flush()

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

def send_IPs_to_PSI (): 
    context = zmq.Context()
    socket = context.socket(zmq.REQ)

    print("Connecting to server...")
    socket.connect("tcp://128.2.204.249:40001")   # bree
    time.sleep(1)

    request = json.dumps({"sensorVideoText":"tcp://128.2.220.118:40003", "sensorAudio": "tcp://128.2.212.138:40001", "sensorDOA": "tcp://128.2.212.138:40002", "sensorVAD": "tcp://128.2.212.138:40003"})   # erebor"
    # request = "tcp://128.2.149.108:40003"
    # request = "tcp://23.227.148.141:40003"
    # "sensorVideoText":"tcp://128.2.212.138:40000" -- Nano
    # "sensorVideoText":"tcp://128.2.220.118:40003" -- erebor

    # Send the request
    payload = {}
    payload['message'] = request
    payload['originatingTime'] = datetime.datetime.utcnow().isoformat()
    print(f"Sending request: {request}")
    socket.send_string(request)

    #  Get the reply
    reply = socket.recv()
    print(f"Received reply: {reply}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='./output/', help='Name of output directory, must be created before running')
    parser.add_argument('--num_channels', default=6, type=int, help='Number of channels being recorded by microphone')
    parser.add_argument('--outfile', default='video', help='Name of video folder for output')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize frames while recording, cannot do in headless mode')
    args = parser.parse_args()
    main(args)