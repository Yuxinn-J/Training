import socket  # 网络连接专用包

HOST = '39.156.69.79'
PORT = 80  # Default port of HTTP


def main():
	# AF_INET IPv4
	# SOCK_STREAM 使用TCP连接
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	# Get local IP and Port from OS
	s.connect((HOST, PORT))
	# bytes, string
	s.send(b'GET / HTTP/1.1\n')
	# Header
	s.send(b'Host: baidu.com\n')
	s.send(b'\n')
	# Body

	# End
	s.send(b'\n')

	# bytes
	data = s.recv(1024)
	# bytes.decode() <--> string.encode()
	print(data.decode())
	# 回收 socket 资源
	s.close()


if __name__ == '__main__':
	main()