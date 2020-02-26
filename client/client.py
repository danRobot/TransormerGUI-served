import socket

s = socket.socket()
host = '192.168.0.10'
port = 8000

s.connect((host, port))
while True: 
    print("From Server: ", s.recv(1024))  #This gets printed after sometime
    s.send("Client please type: ".encode())

s.close()  
