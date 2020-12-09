import socket

HOST = socket.gethostbyname(socket.gethostname())
PORT = 9000


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    print('Server started')
    s.listen()

    while True:
        # 每次接收到 request 都是一个新的 connection
        # addr 客户端地址
        connection, addr = s.accept()
        print('-----------------')
        print(f'connected with: {addr}')
        data = connection.recv(1024)
        print(data.decode())
        connection.send(b'HTTP/1.1 200 OK\n')
        connection.send(b'\n')
        connection.send(b'<h1>Hello World!<h1>\n')
        connection.close()


if __name__ == '__main__':
    main()