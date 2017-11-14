#author: matthew smith mrs9107@g.rit.edu
#file: multithreaded http socket server

from http.server import BaseHTTPRequestHandler
from datetime import datetime
import io
import socket
import threading
import time
import sys
import os

class server:
    default_version = "HTTP/0.9"
    content_type_text = "text/html"
    enable_threading = False
    running_path = os.path.dirname(os.path.abspath(__file__))

    #initialize server socket
    def __init__(self, host, port):
        self.host = host
        self.port = port
        #create tcp/ip socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))

        self.handlers = { "GET": lambda x: self.HTTP_GET(x)}

        self.codes = {    "404": lambda x:self.HTTP_404(x),
                          "403": lambda x:self.HTTP_403(x),
                          "501": lambda x:self.HTTP_501(x)
                          }

    def listen(self):
        self.sock.listen(5) #backlog num, # of connections allowed to be queued

        while True:
            client, address = self.sock.accept()
            client.settimeout(60) #1 minute timeout
            if(self.enable_threading):
                threading.Thread(target = self.handle_client, args= (client, address)).start()
            else:
                self.handle_client(client, address)

    def handle_client(self, client, address):
        size = 1024
        while True:
            try:
                data = client.recv(size)
                print("gotsomething")
                if data:
                    #print(data)
                    response = self.handle_HTTP(data)
                    client.send(response)
                    client.close()
                else:
                    raise error('client disconnected')
                    print('cient disconnected')

            except:
                client.close()
                return False

        return True


    def handle_HTTP(self, data):
        request = HTTP_request(data)
        command = request.command

        if not command in self.handlers.keys():
            return self.codes["501"](request)
        else:
            return self.handlers[command](request)


    def HTTP_GET(self, request):
        if(request.path == '/'):
            file_size = os.path.getsize("hello.htm")
            file_name = "hello.htm"
        else:
            path = self.running_path + request.path
            print(path)
            if(os.path.exists(path)):
                file_size = os.path.getsize(path)
                file_name = request.path[1:]
            else:
                print(":it doesnt exist")
                return self.codes["404"](request)

        response = self.construct_header("200 OK",self.content_type_text, file_size )
        response = response + "\r\n" + (open(file_name).read())
        print("constructed and sending response")
        return bytes(response, "utf8")

    def construct_header(self,response_status, content_type, content_length):
        time = 0
        #time = datetime.now().strftime('%b %d  %I:%M:%S\r\n')
        http_response = ("HTTP/1.1" + response_status + "\r\n" + \
                         "Server: python-custom\r\n" +\
                         "Content-Length: " + str(content_length) + "\r\n" + \
                         "Content-Type: " + content_type + "\r\n" + \
                         "Connection: Closed\r\n" )

        return http_response


    def HTTP_501(self, request):
        construct_header("501 not implemented", content_type_text, 0)
        return 0

    def HTTP_404(self, request):
        file_size = os.path.getsize("404.htm")
        response = self.construct_header("404",self.content_type_text, file_size )
        response = response + "\r\n" + (open("404.htm").read())
        return bytes(response, "utf8")

    def HTTP_403(self, request):
        response = self.construct_header("403",self.content_type_text, file_size )
        response = response + "\r\n" + (open("403.htm").read())
        return 0

#executive decision: project not about text parsing, so offload parsing
#to subset of HTTP library
#https://stackoverflow.com/questions/4685217/parse-raw-http-headers
class HTTP_request(BaseHTTPRequestHandler):

    def __init__(self, request):
        self.rfile = io.BytesIO(request)
        self.raw_requestline = self.rfile.readline()
        self.error_code = self.error_message = None
        self.parse_request()

    def send_error(self, code, message):
        self.error_code = code
        self.error_message = message


def main():
    port = 8080 #default http port
    server('', port).listen()


if __name__ == "__main__":
    main()
