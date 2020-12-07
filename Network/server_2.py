import socket

HOST = socket.gethostbyname(socket.gethostname())
PORT = 9000


def parse_request(request: str):
    line = request.split('\n')
    print('First line of request', lines[0])
    parameters = lines[0].split(' ')
    method, path = parameters[0], parameters[1]
    return f'<h1>This is {method} Method.</h1>'


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
        result = parse_request(data.decode())

        connection.send(b'HTTP/1.1 200 OK\n')
        connection.send(b'\n')
        connection.send(result.encode()
        connection.close()


if __name__ == '__main__':
    main()