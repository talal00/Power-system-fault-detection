import socket

def send_value(value):
    host = '127.0.0.1'
    port = 12345

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # Convert list to string and send
    value_str = ','.join(map(str, value)) 
    client_socket.send(value_str.encode('utf-8'))

    client_socket.close()

if __name__ == "__main__":
    t = [0, 1, 2, 3, 4, 5, 6, 7]
    print(t)
    send_value(t)
