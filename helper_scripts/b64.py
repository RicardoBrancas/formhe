import base64
import sys

file = sys.argv[1]

with open(file) as f:
    print(base64.b64encode(f.read().encode()).decode())