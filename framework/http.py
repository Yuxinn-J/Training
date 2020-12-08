class Request:

    def __init__(self, raw_data: str):
        self.raw_data = raw_data

        lines = raw_data.split('\n')
        items = lines[0].split(' ')
        self.method, self.path = items[0], items[1]


class Response:

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.headers = b'\n'

        if status_code == 200:
            self.first_line = b'HTTP/1.1 200 OK\n'
            self.body = body.encode() + b'\n'
        elif status_code == 302:
            self.first_line = b'HTTP/1.1 302 Found\n'
            self.body = body.encode() + b'Location: http://localhost:9000/\n'

    def to_raw(self) -> bytes:
        result = self.first_line + self.headers + self.body
        return result

