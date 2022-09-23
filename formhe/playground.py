import signal
import time

from formhe.asp.synthesis.ASPSpecGenerator import ASPSpecGenerator
from formhe.asp.instance import Instance
from formhe.trinity.z3_enumerator import Z3Enumerator

instance = Instance('generated_instances/nqueens/0.lp')

spec_generator = ASPSpecGenerator(instance, 2)
trinity_spec = spec_generator.trinity_spec

signal.signal(signal.SIGINT, signal.default_int_handler)

x = 0
start = time.time()
for i in range(2, 4):
    enumerator = Z3Enumerator(trinity_spec, i)
    while prog := enumerator.next():
        x += 1
        print(prog)
        enumerator.update()
        if x == 2000:
            break
    if x == 2000:
        break

print(x)
elapsed = time.time() - start
print('Took', elapsed)
print(x / elapsed, 'enums/s')
