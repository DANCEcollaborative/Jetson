from zmq_utils import *


def find_my_ip(api_service_ip="https://api.ipify.org"):
    my_ip = get(api_service_ip).content.decode("utf8")
    return my_ip


def share_my_ip_with_psi(my_ip, remote_port=40000, audio_port=40001, doa_port=40002, vad_port=40003, psi_server_ip_port="tcp://128.2.204.249:40001"):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    print("Connecting to server...")
    socket.connect(psi_server_ip_port)   # bree
    # request = f"tcp://{my_ip}:{my_port}"     # erebor
    # request = json.dumps({"sensorVideoText":"tcp://128.2.212.138:40000", "sensorAudio": "tcp://128.2.212.138:40001", "sensorDOA": "tcp://128.2.212.138:40002", "sensorVAD": "tcp://128.2.212.138:40003"})   # erebor"
    request = json.dumps({"remoteIP":f"tcp://{my_ip}:{remote_port}", "audio_channel": f"tcp://{my_ip}:{audio_port}", "doa": f"tcp://{my_ip}:{doa_port}", "vad": f"tcp://{my_ip}:{vad_port}"})   # erebor"

    payload = {}
    payload["message"] = request
    payload["originatingTime"] = datetime.datetime.utcnow().isoformat()
    print(f"Sending request: {request}")
    socket.send_string(request)
    print(f"Waiting for reply...")
    reply = socket.recv()
    print(f"Received reply: {reply}")


if __name__ == "__main__":
    bree_ip = "128.2.204.249"
    psi_server_ip_port=f"tcp://{bree_ip}:40001"
    # my_ip = find_my_ip()
    my_ip = "172.26.160.201"
    print('My public IP address is: {}'.format(my_ip))
    share_my_ip_with_psi(my_ip, psi_server_ip_port=psi_server_ip_port)

