import sys
import sass

filename = sys.argv[1]

kernels = sass.parse_ncu_sass(filename)

k : sass.Kernel
for k in kernels.values():
    print(k.name, k.ncalls, len(k.trace))
