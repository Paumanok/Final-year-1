#author: matthew smith mrs9107@g.rit.edu
#file: multithreaded http socket server

from http.server import BaseHTTPRequestHandler
from  io import StringIO
from datetime import datetime
import socket
import threading


class server:
    default_version = "HTTP/0.9"
    content_type_text = "text/html; charset=iso-8859-1"

    #initialize server socket
    def __init__(self, host, port):
        self.host = host
        self.port = port
        #create tcp/ip socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))

        self.handlers = { "GET": lambda:self.HTTP_GET(request)}

        self.codes = {    "404": lambda:self.HTTP_404(request),
                          "403": lambda:self.HTTP_403(request),
                          "501": lambda:self.HTTP_501(request)
                          }

    def listen(self):
        self.sock.listen(5) #backlog num, # of connections allowed to be queued

        while True:
            client, address = self.sock.accept()
            client.settimeout(60) #1 minute timeout
            threading.Thread(target = self.handle_client, args= (client, address)).start()

    def handle_client(self, client, address):
        size = 1024
        while True:
            try:
                #get data
                data = client.recv(size)
                if data:
                    print(data)
                    # Reply as HTTP/1.1 server, saying "HTTP OK" (code 200).
                    #response_proto = 'HTTP/1.1'
                    #response_status = '200'
                    #response_status_text = 'OK' # this can be random
                    #client.send('%s %s %s' % (response_proto, response_status, \
                    #                                    response_status_text))
                    #handle_HTTP(data)
                else:
                    raise error('client disconnected')

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

        return 0

    def construct_header(self,response_status, content_type, content_length):
        time = datetime.now().strftime('%b %d  %I:%M:%S\n')
        http_response = ("HTTP/1.1" + response_status + "\r\n" + \
                         "Date: " + time + "\r\n" + \
                         "Server: python-custom\r\n" +\
                         "Content-Length: " + str(content_length) + "\r\n" + \
                         "Connection: Closed\r\n" + \
                         "Content-Type: " + content_type + "\r\n" )
        return http_response


    def HTTP_501(self, request):
        construct_header("500 not implemented", content_type_text, 0)
        return 0

    def HTTP_404(self, request):
        return 0

    def HTTP_401(self, request):
        return 0

#executive decision: project not about text parsing, so offload parsing
#to subset of HTTP library
#https://stackoverflow.com/questions/4685217/parse-raw-http-headers
class HTTP_request(BaseHTTPRequestHandler):

    def __init__(self, request):
        self.raw_file = StringIO(request)
        self.raw_request = self.raw_file.readline()
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
