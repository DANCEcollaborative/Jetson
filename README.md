To clone recursively all the submodules run:
`git clone --recurse-submodules https://github.com/sohamtiwari3120/jetson.git`

In terminal run:
1. `cd jetson/`
2. `bash setup_mic_support.sh`. You will have to enter sudo password.
3. `bash update_mic_firmware.sh -a` to update firmware for 6 channel support.
4. `sudo python3 record_audio.py` to start audio recording. Take note of the output folder in `output/audio/<mm-dd-yy_HH_MM_SS>/` to remember which folder contains the correct audios when sharing it with others. REMEMBER: Only press CTRL+C to stop audio recording in the terminal. If you do not do that then the recorded audio will not be saved. 