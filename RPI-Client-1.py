import socket
import numpy as np
from multiprocessing import Process

def send_value(host, port, value):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # Convert NumPy array to bytes and send
    value_bytes = value.tobytes()
    client_socket.send(value_bytes)

    client_socket.close()

if __name__ == "__main__":
    t1 = np.array([[1, 2, 3, 4, 5, 6],
                   [3, 10, 5, 6, 70, 1],
                   [4, 11, 15, 16, 76, 4],
                   [0, 0, 0, 0, 0, 1],
                   [20, 12, 25, 15, 13, 0]], dtype=int)

    t2 = np.array([[9, 8, 7, 6, 5, 4],
                   [3, 7, 5, 1, 9, 2],
                   [1, 5, 3, 8, 2, 6],
                   [0, 1, 0, 1, 0, 1],
                   [3, 4, 5, 6, 7, 8]], dtype=int)

    print("Array 1:")
    print(t1)
    print("Array 2:")
    print(t2)

    host1 = '127.0.0.1'
    port1 = 12345

    host2 = '127.0.0.1'
    port2 = 12345

    process1 = Process(target=send_value, args=(host1, port1, t1))
    process2 = Process(target=send_value, args=(host2, port2, t2))

    process1.start()
    process2.start()

    process1.join()
    process2.join()
