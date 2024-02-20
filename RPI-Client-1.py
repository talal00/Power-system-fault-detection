import pandas as pd 
import socket
import numpy as np

data = pd.read_csv('dataset.csv', sep=';').fillna(0)

def send_value(value):
    host = '127.0.0.1'
    port = 12345

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # Convert NumPy array to bytes and send
    value_bytes = value.tobytes()
    client_socket.send(value_bytes)

    client_socket.close()

if __name__ == "__main__":
    t = np.array([[1, 2, 3, 4, 5, 6],
                  [3, 10, 5, 6, 70, 1],
                  [4, 11, 15, 16, 76, 4],
                  [0, 0, 0, 0, 0, 1],
                  [20, 12, 25, 15, 13, 0]], dtype=int)

    print(t)
    send_value(t)
