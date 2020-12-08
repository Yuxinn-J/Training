import socket
import ssl

# 2 ** 8 ** 4
HOST = 'movie.douban.com'
# Default port of HTTP
# PORT = 80
# Default port of HTTPS
PORT = 443


def main():
    # IPv4
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s = ssl.wrap_socket(s)
    # Get local IP and Port from OS
    s.connect((HOST, PORT))
    s.send(b'GET /top250 HTTP/1.1\n')
    # Header
    s.send(f'Host: {HOST}\n'.encode())
    s.send(f'''
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) 
Chrome/86.0.4240.111 Safari/537.36 Edg/86.0.622.51'''.encode())
    s.send(b'\n')
    s.send(b'\n')
    print('Message sent')
    # Body

    # bytes
    data = b''
    while True:
        chuck = s.recv(1024)
        if len(chuck) < 1024:
            data += chuck
            break
        data += chuck

    data = data.decode()
    print(data)
    for i in range(1, 25 + 1):
        index = data.find(f'<em class="">{i}</em')
        items = data[index:index + 300].split('"')
        print(items[7])

    # Close connection
    s.close()


if __name__ == '__main__':
    main()