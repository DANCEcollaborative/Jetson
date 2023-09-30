import sys

required_import_paths = [
    "~/",
    "/usr/local/lib/python3.6/pyrealsense2",
    "~/.local/lib/python3.6/site-packages",
]
sys.path = sys.path + required_import_paths

# Jetson
jetson_ip = "128.2.212.138"
remoteIp_port = 60000
audio_port = 60001
doa_port = 60002
vad_port = 60003

confusion_classifier_res_port = 61001

# BREE
bree_ip = "128.2.204.249"
bree_initial_response_port = 40001
