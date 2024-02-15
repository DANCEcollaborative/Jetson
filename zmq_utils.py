"""
ZeroMQ utility functions
"""

import zmq, datetime, msgpack, time, json
from requests import get

def generate_current_dotnet_datetime_ticks(base_time = datetime.datetime(1, 1, 1)):
    return (datetime.datetime.utcnow() - base_time)/datetime.timedelta(microseconds=1) * 1e1

def send_payload(pub_sock, topic, message, originatingTime=None):
    payload = {}
    payload[u"message"] = message
    if originatingTime is None:
        originatingTime = generate_current_dotnet_datetime_ticks()
    payload[u"originatingTime"] = originatingTime
    pub_sock.send_multipart([topic.encode(), msgpack.dumps(payload)])
    return originatingTime

def create_socket(ip_address='tcp://*:40003'):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(ip_address)
    return socket