import socket
import numpy as np
import pandas as pd
import multiprocessing

def receive_array(host, port, queue):
    while True:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        
        print(f"Connected to {host}:{port}")

        data = client_socket.recv(4096)
        if not data:
            client_socket.close()
            continue

        # Convert received bytes back to NumPy array
        received_array = pd.DataFrame(np.frombuffer(data, dtype=int).reshape(-1, 6), columns=['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc'])

        print("Received array:")
        print(received_array)

        client_socket.close()

        # Put the received array into the queue
        queue.put(received_array)

def send_array(host, port, queue):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"Listening for connections on {host}:{port}")

    conn, addr = server_socket.accept()
    print(f"Received connection from {addr}")

    while True:
        # Get array from the queue
        array = queue.get()

        # Convert NumPy array to bytes and send
        array_bytes = array.tobytes()
        conn.send(array_bytes)

    conn.close()
    server_socket.close()

if __name__ == "__main__":
    # Host and port for sending the received array
    send_host = '127.0.0.1'
    send_port = 12346

    # Host and port for receiving the array
    receive_host = '127.0.0.1'
    receive_port = 12345

    # Create a queue for communication between processes
    queue = multiprocessing.Queue()

    # Create two processes for receiving and sending arrays
    receive_process = multiprocessing.Process(target=receive_array, args=(receive_host, receive_port, queue))
    send_process = multiprocessing.Process(target=send_array, args=(send_host, send_port, queue))

    # Start both processes
    receive_process.start()
    send_process.start()

    # Join both processes to wait for their completion
    receive_process.join()
    send_process.join()
