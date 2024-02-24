import socket
import numpy as np

def send_array(host, port, array):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # Convert NumPy array to bytes and send
    array_bytes = array.tobytes()
    client_socket.send(array_bytes)

    client_socket.close()

if __name__ == "__main__":
    # Sample array to send
    sample_array = np.array([[9, 8, 7, 6, 5, 4],
                   [3, 7, 5, 1, 9, 2],
                   [1, 5, 3, 8, 2, 6],
                   [0, 1, 0, 1, 0, 1],
                   [3, 4, 5, 6, 7, 8]], dtype=int)


    # Host and port to send the array data
    host = '127.0.0.1'  
    port = 12345        

    # Send the array data
    send_array(host, port, sample_array)
