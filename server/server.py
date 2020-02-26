import socket

s = socket.socket()         # Create a socket object
host = '192.168.0.10'    #private ip address of machine running fedora
port = 8000
s.bind((host, port))

s.listen(5)
c, addr = s.accept()
print('Got connection from', addr)    #this line never gets printed
while True:
   c.send("Server please type: ".encode())
   print("From Client: ", c.recv(1024))

c.close()
