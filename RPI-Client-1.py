import socket
import sys

def send_value(value):
    host = '127.0.0.1'
    port = 12345

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    client_socket.send(value.encode('utf-8'))

    client_socket.close()

if __name__ == "__main__":
    #Argument = ' '.join(sys.argv[1:])

    #value_to_send = Argument
    send_value("00")
