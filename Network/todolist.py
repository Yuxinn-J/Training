import socket

HOST = socket.gethostbyname(socket.gethostname())
PORT = 9000


def get_method_and_path(request: str):
    line = request.split('\n')
    parameters = line[0].split(' ')
    method, path = parameters[0], parameters[1]
    return method, path


def read_data():
    with open('data.txt') as f:
        data = f.read()
        items = data.split('\n')

    todo_list = []
    for item in items:
        if ':' not in item:
            continue
        todo = {}
        item_id, done, value = item.split(':')
        todo['id'] = item_id
        todo['done'] = bool(int(done))
        todo['value'] = value
        todo_list.append(todo)

    return todo_list


def parse_request(request: str):
    method, path = get_method_and_path(request)

    if path == '/':
        todo_list = read_data()
        response = '<h1>Todo list</h1>\n'
        for todo in todo_list:
            response += f'<form method="POST" action="/todo/{todo["id"]}">{todo["value"]}<input type="submit"/></form>'
        return response, 200

    if path.startswith('/todo'):
        item_id = path.split('/')[-1]
        if item_id.isdigit():
            item_id = int(item_id)
            todo_list = read_data()
            return f'<p>{todo_list[item_id]}</p>', 302
        return '<h1>404 not found</h1>', 200


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    print('Server started')
    s.listen()

    while True:
        connection, addr = s.accept()
        print('-----------------')
        print(f'connected with: {addr}')
        data = connection.recv(1024)
        print(data.decode())
        result, code = parse_request(data.decode())

        if code == 200:
            connection.send(b'HTTP/1.1 200 OK\n')
            connection.send(b'\n')
        elif code == 302:
            connection.send(b'HTTP/1.1 302 Found\n')
            connection.send(b'Location: http://localhost:9000/\n')
        connection.send(b'\n')
        connection.send(result.encode())
        connection.close()


if __name__ == '__main__':
    main()