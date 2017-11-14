#author: matthew smith mrs9107@g.rit.edu
#file: multithreaded http socket client

import socket
import sys
import threading

def connect():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    #max threads allowed
    max_threads = 20

    # Connect the socket to the port where the server is listening
    server_address = ('localhost', 8080)
    print('connecting to {} port {}'.format(*server_address))
    #sock.connect(server_address)

    request_get = "GET /taleoftwocities.htm HTTP/1.1\r\nHost: localhost:8080\r\nUser-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:56.0) Gecko/20100101 Firefox/56.0\r\nAccept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\nAccept-Language: en-US,en;q=0.5\r\nAccept-Encoding: gzip, deflate\r\nDNT: 1\r\nConnection: keep-alive\r\nUpgrade-Insecure-Requests: 1\r\n\r\n"

    try:
        while True:
            # Send data
            sock.connect(server_address)
            message = bytes(request_get, "utf8")
            print('sending {!r}'.format(message))
            sock.send(message)
            sock.close()

            # Look for the response
            amount_received = 0
            amount_expected = len(message)

            while amount_received < amount_expected:
                data = sock.recv(48)
                amount_received += len(data)
                print('received {!r}'.format(data))

    finally:
        print('closing socket')
        sock.close()

connect()
