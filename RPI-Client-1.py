import socket
import numpy as np
import pandas as pd

def receive_array(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"Listening for array data on {host}:{port}")

    conn, addr = server_socket.accept()
    print(f"Received connection from {addr}")

    data = conn.recv(4096)
    if not data:
        return None

    # Convert received bytes back to NumPy array
    received_array = pd.DataFrame(np.frombuffer(data, dtype=int).reshape(-1, 6), columns=['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc'])

    print("Received array:")
    print(received_array)

    conn.close()
    server_socket.close()

    return received_array

def send_array(host, port, array):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # Convert NumPy array to bytes and send
    array_bytes = array.tobytes()
    client_socket.send(array_bytes)

    client_socket.close()

if __name__ == "__main__":
    # Host and port for receiving the array
    receive_host = '127.0.0.1'
    receive_port = 12345

    # Host and port for sending the received array
    send_host = '127.0.0.1'
    send_port = 12346

    while True:
        # Receive the array
        received_array = receive_array(receive_host, receive_port)

        if received_array is not None:
            # Send the received array
            send_array(send_host, send_port, received_array)
