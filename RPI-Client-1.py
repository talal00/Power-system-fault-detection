import socket
import pandas as pd  

data = pd.read_csv('dataset.csv',sep=';').fillna(0)



def send_value(value):
    host = '127.0.0.1'
    port = 12345

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    client_socket.send(value.encode('utf-8'))

    client_socket.close()

if __name__ == "__main__":
    X=data.iloc[:,1:7]
    X.describe()
    Y=data.iloc[:,0]
    Y.describe()
    #Argument = ' '.join(sys.argv[1:])

    #value_to_send = Argument
    print (X)
    #send_value(X)
