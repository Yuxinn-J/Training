"""
Model: database
View: template
Controller
"""
import socket
from framework.http import Request
from framework.http import Response

HOST = socket.gethostbyname(socket.gethostname())
PORT = 9000


def index(request: Request) -> Response:
    # Todos
    with open('template.txt') as f:
        template = f.read()
        template.format(data1='Register Courses')

    return Response(200, template)


def mark(request: Request) -> Response:
    with open('template.txt') as f:
        item_id = request.path[7:-1]
        template = f.read()
        template.format(data1='Done: Register Courses')
    return Response(302, template)


def dispatch_request(data: bytes) -> bytes:
    request = Request(data.decode())
    # 根据不同path执行不同function
    routes = {
        '/': index,
        '/todo/': mark,
    }
    controller = routes.get(request.path)
    if controller:
        response = controller(request)
        return Response.to_raw(response)
    else:
        return b'<h1>404 not found</h1>'


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    print('Server started')
    s.listen()

    while True:
        connection, addr = s.accept()
        try:
            print('----------')
            print(f'connected with: {addr}')
            data = connection.recv(1024)
            print(data.decode())
            send_data = dispatch_request(data)
            connection.send(send_data)
        except Exception as e:
            print(e)
        finally:
            connection.close()


if __name__ == '__main__':
    main()