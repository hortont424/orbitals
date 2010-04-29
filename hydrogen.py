from math import *
from time import time
import pyopencl as cl
import numpy
from PIL import Image

mf = cl.mem_flags

ctx = cl.Context(dev_type=cl.device_type.GPU)
print ctx.get_info(cl.context_info.DEVICES)
queue = cl.CommandQueue(ctx)

# There's a strange bug where changing resolution sometimes makes no output
imageResolution = 400

res = numpy.int32(imageResolution)
pointCount = imageResolution ** 2
n = numpy.int32(3)
l = numpy.int32(2)
m = numpy.int32(0)

output = numpy.zeros(pointCount).astype(numpy.float32)
outputBuffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=output)

kernelFile = open('hydrogen.cl', 'r')
prg = cl.Program(ctx, kernelFile.read()).build()
kernelFile.close()

def independentPsi(n, l):
    a = 1.0
    rootFirst = (2.0 / (n * a)) ** 3
    rootSecond = factorial(n - l - 1.0) / (2.0 * ((n * factorial(n + l)) ** 3))

    return sqrt(rootFirst * rootSecond)

def doDensity():
    ipsi = numpy.float32(independentPsi(n, l))
    prg.density(queue, [pointCount], ipsi, n, l, m, outputBuffer, res)
    cl.enqueue_read_buffer(queue, outputBuffer, output).wait()

before = time()
doDensity()
print time() - before

scaleFactor = 255.0 / max(output)
for i in range(0, len(output)):
    output[i] *= scaleFactor

img = Image.new("L", (res, res))
img.putdata(output)
img.show()
img.save("/tmp/orbitals.png", "PNG")

outputBuffer.release()
