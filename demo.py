from math import *
import pyopencl as cl
import numpy

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

pointCount = 100
n = numpy.float32(1.0)
l = numpy.float32(0.0)
m = numpy.float32(0.0)

output = numpy.zeros(pointCount).astype(numpy.float32)
dest_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)

prg = cl.Program(ctx, """
__kernel void density(__global float ipsi, __global float n, __global float l,
                      __global float m, __global float * output)
{
    int gid = get_global_id(0);

    float x, y, z, theta, phi, r;

    output[gid] = ipsi;
}
""").build()

def independentPsi(n, l):
    a = 1.0
    rootFirst = (2.0 / (n * a)) ** 3
    rootSecond = factorial(n - l - 1.0) / (2.0 * ((n * factorial(n + l)) ** 3))

    return sqrt(rootFirst * rootSecond)

ipsi = numpy.float32(independentPsi(n, l))
prg.density(queue, [pointCount], ipsi, n, l, m, dest_buf)
cl.enqueue_read_buffer(queue, dest_buf, output).wait()
print output