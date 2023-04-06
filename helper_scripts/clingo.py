import subprocess
import sys

subprocess.run(['clingo'] + sys.argv[1:])