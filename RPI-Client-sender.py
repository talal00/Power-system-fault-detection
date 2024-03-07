import socket
import numpy as np
import pandas as pd
import multiprocessing

def receive_array(host, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print ("Opening Socket Server")
    #client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #client_socket.bind((host, port))
    #client_socket.listen(1)
    #conn, addr = client_socket.accept()
    i=0
    while i==0:

        print(f"Connected to {host}:{port}")

        try:

            data = client_socket.recv(4096)
            if not data:
                client_socket.close()
                continue

            # Convert received bytes back to NumPy array
            #received_array = pd.DataFrame(np.frombuffer(data, dtype=int).reshape(-1, 6), columns=['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc'])

            print("Received array:")
            #print(received_array)
            print(data)
            i=1
            client_socket.close()

            # Put the received array into the queue
            #queue.put(received_array)
            #queue.put(data)

        except KeyboardInterrupt:
            print("Closing client socket")
            client_socket.close()
            break
    return data

def send_array(host, port, data):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.connect((host, port))

    print(f"Listening for connections on {host}:{port}")

    # Convert NumPy array to bytes and send
    array = np.frombuffer(data, dtype=int) #.toBytes()
    print("Predicted Data", array)

    for value in array:
        if value == 1:
        # Your code to run when the value is 1
            print("Fault Detected!")
            # Create a NumPy variable with a 32-bit integer value of 1
            integer_to_send = np.array([1], dtype=np.int32)
            # Convert the NumPy array to bytes
            byte_data = integer_to_send.tobytes()
    
   #conn, addr = server_socket.accept()
   # print(f"Received connection from {addr}")
            try:
                while True:
        # Get array from the queue
            #array = queue.get()
                    server_socket.send(byte_data)
            except KeyboardInterrupt:
                break
                print("Closing sending socket")
                server_socket.close()
#        else
 #           while True:
  #              integer_to_send = np.array([0], dtype=np.int32)
                # Convert the NumPy array to bytes
    #            byte_data = integer_to_send.tobytes()
   #             server_socket.send(byte_data)

if __name__ == "__main__":
    # Host and port for sending the received array
    send_host = '193.166.118.230'
    send_port = 7200

    # Host and port for receiving the array
    receive_host = '192.168.2.1'
    receive_port = 12347

    # Create a queue for communication between processes
    #queue = multiprocessing.Queue()

    # Create two processes for receiving and sending arrays
    #receive_process = multiprocessing.Process(target=receive_array, args=(receive_host, receive_port, queue))
    #send_process = multiprocessing.Process(target=send_array, args=(send_host, send_port, queue))
    try:
        data = receive_array(receive_host, receive_port)
        send_array(send_host, send_port, data)
     #Start both processes
     #   receive_process.start()
        #send_process.start()

    # Join both processes to wait for their completion
        #receive_process.join()
        #send_process.join()
    except KeyboardInterrupt:
        #receive_process.terminate()
        #send_process.terminate()
        #receive_process.join()
        #send_process.join()
        client_socket.close()
        server_socket.close()
