import socket
import numpy as np
import pandas as pd
import multiprocessing

def receive_array(host, port, queue):

    while True:
        try:

            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((host, port))

            print(f"Connected to {host}:{port}")

            data = client_socket.recv(1024)
            if not data:
                client_socket.close()
                continue

            # Convert received bytes back to NumPy array
            received_array = data #np.frombuffer(data, dtype=float).reshape(-1, 6) #, columns=['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc'])
            numpy_array=np.frombuffer(data, dtype=np.int32).reshape(-1, 6) #, columns=['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc'])
            print("Numpy_array", numpy_array)


            print("Received array:")
            print(type(received_array))
            print(received_array)
            client_socket.close()

            # Put the received array into the queue
            #queue.put(received_array)
            queue.put(received_array)

        except KeyboardInterrupt:
            pass
        finally:
            print("Closing client socket")
            client_socket.close()

def send_array(host, port, queue):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.connect((host, port))

    #server_socket.bind((host, port))
    #server_socket.listen(1)

    print(f"Connection established on {host}:{port}")

    #conn, addr = server_socket.accept()
   # print(f"Received connection from {addr}")
    try:
        while True:
        # Get array from the queue
            array = queue.get()
            print("Array: ", array)
            print("type of array", type(array))
        # Convert NumPy array to bytes and send
            array_bytes = array #.tobytes()
            server_socket.send(array_bytes)
    except KeyboardInterrupt:
        pass
    finally:
            print("Closing sending socket")
            server_socket.close()

if __name__ == "__main__":
    # Host and port for sending the received array
    send_host = '192.168.2.1'
    send_port = 12345

    # Host and port for receiving the array
    receive_host = '193.166.118.230'
    receive_port = 7200

    # Create a queue for communication between processes
    queue = multiprocessing.Queue()

    # Create two processes for receiving and sending arrays
    receive_process = multiprocessing.Process(target=receive_array, args=(receive_host, receive_port, queue))
    send_process = multiprocessing.Process(target=send_array, args=(send_host, send_port, queue))
    try:
    # Start both processes
        receive_process.start()
        send_process.start()

    # Join both processes to wait for their completion
        receive_process.join()
        send_process.join()
    except KeyboardInterrupt:
        receive_process.terminate()
        send_process.terminate()
        receive_process.join()
        send_process.join()
