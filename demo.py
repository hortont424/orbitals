from math import *
import pyopencl as cl
import numpy

ctx = cl.Context(dev_type=cl.device_type.GPU)
print ctx.get_info(cl.context_info.DEVICES)
queue = cl.CommandQueue(ctx)

pointCount = 100
n = numpy.int32(1)
l = numpy.int32(0)
m = numpy.int32(0)

xyz = 6 * numpy.random.rand(3 * pointCount).astype(numpy.float32) - 3
output = numpy.zeros(pointCount).astype(numpy.float32)

mf = cl.mem_flags
xyz_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xyz)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)

prg = cl.Program(ctx, """
int L(int a, int b, int x)
{
    if(a == 0)
    {
        if(b == 0)
            return 1;
        else if(b == 1)
            return 1 - x;
        else if(b == 2)
            return 2 - (4 * x) + (x * x);
        else if(b == 3)
            return 6 - (18 * x) + (9 * x * x) - (x * x * x);
    }
    else if(a == 1)
    {
        if(b == 0)
            return 1;
        else if(b == 1)
            return 2 * (2 - x);
        else if(b == 2)
            return 3 * (6 - (6 * x) + (x * x));
        else if(b == 3)
            return 4 * (24 - (36 * x) + (12 * x * x) - (x * x * x));
    }
    else if(a == 2)
    {
        if(b == 0)
            return 2;
        else if(b == 1)
            return 6 * (3 - x);
        else if(b == 2)
            return 12 * (12 - (8 * x) + (x * x));
        else if(b == 3)
            return 20 * (60 - (60 * x) + (15 * x * x) - (x * x * x));
    }
    else if(a == 3)
    {
        if(b == 0)
            return 6;
        else if(b == 1)
            return 24 * (4 - x);
        else if(b == 2)
            return 60 * (20 - (10 * x) + (x * x));
        else if(b == 3)
            return 120 * (120 - (90 * x) + (18 * x * x) - (x * x * x));
    }
}

float P(int a, int b, float x)
{
    if(a == 0)
    {
        if(b == 0)
            return 1.0;
        else if(b == 1)
            return x;
        else if(b == 2)
            return 0.5 * (-1 + (3.0 * x * x));
        else if(b == 3)
            return 0.5 * ((-3.0 * x) + (5.0 * x * x * x));
    }
    else if(a == 1)
    {
        if(b == 0)
            return 0.0;
        else if(b == 1)
            return -sqrt(1.0 - (x * x));
        else if(b == 2)
            return (-3.0 * x) * sqrt((float)1.0 - (x * x));
        else if(b == 3)
            return - (3.0 / 2.0) * sqrt((float)1.0 - (x * x)) *
                (-1.0 + (5.0 * x * x));
    }
    else if(a == 2)
    {
        if(b == 0)
            return 0.0;
        else if(b == 1)
            return 0.0;
        else if(b == 2)
            return -3.0 * (-1.0 + (x * x));
        else if(b == 3)
            return -(15.0 * x) * (-1.0 + (x * x));
    }
    else if(a == 3)
    {
        if(b == 0)
            return 0.0;
        else if(b == 1)
            return 0.0;
        else if(b == 2)
            return 0.0;
        else if(b == 3)
            return -15.0 * pow(1.0 - (x * x), (3.0 / 2.0));
    }
}

float EP(float m)
{
    if(m >= 0)
    {
        return 1.0 / m;
    }
    else
    {
        return 1;
    }
}

int fact(int n)
{
    int f = 1;

    /*while (n > 1)
    {
        f = f * n;
        n = n - 1;
    }*/

    return n;
}

float2 cnew(float x, float y)
{
    float2 c;
    c.x = x;
    c.y = y;
    return c;
}

float2 csqrtf(float a)
{
    if(a < 0.0)
        return cnew(0.0, sqrtf(-a));
    else
        return cnew(sqrtf(a), 0.0);
}

float2 cexp(float2 a)
{
    float module = exp(a.x);
    float angle = a.y;
    return cnew(module * native_cos(angle), module * native_sin(angle));
}

float2 cmul(float2 a, float2 b)
{
    return cnew(mad(-a.y, b.y, a.x * b.x), mad(a.y, b.x, a.x * b.y));
}

float2 Y(int m, int l, float theta, float phi)
{
    float rootFirst = (2.0 * l + 1.0) / (4 * 3.1415926);
    float rootSecond = fact(l - abs(m)) / fact(l + abs(m));
    float2 root = csqrtf(rootFirst * rootSecond);
    float2 eiStuff = cmul(cexp(cmul(csqrtf(-1), m * phi)),
                          cnew(P(m, l, native_cos(theta)), 0.0));

    return cnew(rootSecond, rootSecond);//cmul(root, eiStuff);
}

__kernel void density(__global float * xyz, __global float ipsi,
                      __global int n, __global int l,
                      __global int m, __global float * output)
{
    int gid = get_global_id(0);

    float theta, phi, r;
    float4 pos = {0, 0, 0, 0};

    // use vector load?
    pos.x = xyz[(gid * 3) + 0];
    pos.y = xyz[(gid * 3) + 1];
    pos.z = xyz[(gid * 3) + 2];

    // Convert cartesian coordinates to spherical
    r = length(pos);
    theta = acos(pos.z / r);
    phi = atan2(pos.y, pos.x);



    output[gid] = Y(0,0,.3,.5).x;
}
""").build()

def independentPsi(n, l):
    a = 1.0
    rootFirst = (2.0 / (n * a)) ** 3
    rootSecond = factorial(n - l - 1.0) / (2.0 * ((n * factorial(n + l)) ** 3))

    return sqrt(rootFirst * rootSecond)

ipsi = numpy.float32(independentPsi(n, l))
prg.density(queue, [pointCount], xyz_buf, ipsi, n, l, m, dest_buf)
cl.enqueue_read_buffer(queue, dest_buf, output).wait()
print output