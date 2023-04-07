cd usb_4_mic_array

download_channel_6=false
download_channel_1=false

while getopts a:b: flag
do
    case "${flag}" in
        a) download_channel_6=${OPTARG};;
        b) download_channel_1=${OPTARG};;
    esac
done

echo "Download 6 channel firmware (-a) = $download_channel_6"
echo "Download 1 channel firmware (-b) = $download_channel_1"
if [ "$download_channel_6" = true ] ; then
    echo "Downloading 6-channel firmware..."
    sudo python3 dfu.py --download 6_channels_firmware.bin  # The 6 channels version 
# if you want to use 1 channel,then the command should be like:
elif [ "$download_channel_1" = true ] ; then
    echo "Downloading 1-channel firmware..."
    sudo python3 dfu.py --download 1_channel_firmware.bin  # The 1 channel version 
fi