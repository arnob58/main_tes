import http.server
import socketserver
import threading
import time

class ChatHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.lock = threading.Lock()
        self.client_data = []

    def do_POST(self):
        if self.path == '/chat':
            content_length = int(self.headers['Content-Length'])
            message = self.rfile.read(content_length).decode('utf-8')
            with self.lock:
                self.client_data.append(message)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"Message '{message}' received and processed.")

def system_handler(message):
    """Simulates a system handler that processes the message."""
    time.sleep(0.1)
    return f"Processed: {message}"

def run_server(server_class=socketserver.TCPServer, host='localhost', port=8080):
    httpd = server_class((host, port), ChatHandler)
    print(f"Starting httpd server on {host}:{port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
