import zmq, datetime, time, json
from config import *

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)

    print("Connecting to server...")
    socket.connect(f"tcp://{bree_ip}:{bree_initial_response_port}")  # bree
    time.sleep(1)

    # request = "tcp://72.95.139.140:40003"
    request = json.dumps(
        {
            "remoteIP": f"tcp://{jetson_ip}:{remoteIp_port}",
            "audio_channel": f"tcp://{jetson_ip}:{audio_port}",
            "doa": f"tcp://{jetson_ip}:{doa_port}",
            "vad": f"tcp://{jetson_ip}:{vad_port}",
        }
    )  # erebor"
    # request = json.dumps({"sensorVideoText":"tcp://128.2.212.138:40000", "sensorAudio": "tcp://128.2.212.138:40001", "sensorDOA": "tcp://128.2.212.138:40002", "sensorVAD": "tcp://128.2.212.138:40003"})   # erebor"
    # request = "tcp://128.2.149.108:40003"
    # request = "tcp://23.227.148.141:40003"

    # Send the request
    payload = {}
    payload["message"] = request
    payload["originatingTime"] = datetime.datetime.utcnow().isoformat()
    print(f"Sending request: {request}")
    socket.send_string(request)

    #  Get the reply
    reply = socket.recv()
    print(f"Received reply: {reply}")


if __name__ == "__main__":
    main()