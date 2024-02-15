from zmq_utils import *

def find_my_ip(api_service_ip='https://api.ipify.org'):
    my_ip = get(api_service_ip).content.decode('utf8')
    return my_ip

def share_my_ip_with_psi(my_ip, my_port=40003, psi_server_ip_port="tcp://128.2.204.249:40001"):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    print("Connecting to server...")
    socket.connect(psi_server_ip_port)   # bree
    request = f"tcp://{my_ip}:{my_port}"     # erebor
    payload = {}
    payload['message'] = request
    payload['originatingTime'] = datetime.datetime.utcnow().isoformat()
    print(f"Sending request: {request}")
    socket.send_string(request)
    print(f"Waiting for reply...")
    reply = socket.recv()
    print(f"Received reply: {reply}")