from bentoml import HTTPServer

server = HTTPServer("bento-server:formhellm", port=3000)

server.start()
