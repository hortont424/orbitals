#!/usr/bin/env python

import pyopencl as cl
import numpy
import sys

from math import *
from time import time
from PIL import Image
from optparse import OptionParser

def independentPsi(n, l):
    """The first term of psi is constant with respect to the entire image, so
    it can be computed outside of the inner loop."""
    rootFirst = (2.0 / n) ** 3
    rootSecond = factorial(n - l - 1.0) / (2.0 * ((n * factorial(n + l)) ** 3))
    return sqrt(rootFirst * rootSecond)

def renderOrbitals((ni, li, mi), imageResolution):
    """Create various buffers to shuttle data to/from the CPU/GPU, then
    execute the kernel for each pixel in the input image, and copy the
    data into a numpy array"""
    res = numpy.int32(imageResolution)
    pointCount = imageResolution ** 2
    n = numpy.int32(ni)
    l = numpy.int32(li)
    m = numpy.int32(mi)

    # Create native buffer and mirror OpenCL buffer
    mf = cl.mem_flags
    output = numpy.zeros(pointCount).astype(numpy.float32)
    outputBuffer = cl.Buffer(main.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                             hostbuf=output)

    # Evaluate first term of psi once for entire image
    ipsi = numpy.float32(independentPsi(n, l))

    # Evaluate the rest of psi once for each pixel, copy into output buffer
    before = time()
    main.prg.density(main.queue, [pointCount], ipsi, n, l, m, outputBuffer, res).wait()
    computeDuration = time() - before
    cl.enqueue_read_buffer(main.queue, outputBuffer, output).wait()
    copyDuration = time() - before - computeDuration

    outputBuffer.release()

    return (output, (computeDuration, copyDuration))

def exportImage(output):
    """Export a visual representation of the computed density of the
    wavefunction as a PNG (to /tmp/orbitals.png, for now)"""
    # Linearly scale image data so that brightest pixel is white
    scaleFactor = 255.0 / max(output)
    for i in range(0, len(output)):
        output[i] *= scaleFactor

    # Output PNG of computed orbital density
    res = int(sqrt(len(output)))
    img = Image.new("L", (res, res))
    img.putdata(output)
    img.show()
    img.save("/tmp/orbitals.png", "PNG")

def benchmark(skip, params=(3,2,0)):
    """Time computation of the image at various different resolutions"""
    for res in range(100,1010,skip):
        minComputeDuration = 100000000
        minCopyDuration = 100000000

        for itr in range(5):
            (output, (computeDur, copyDur)) = renderOrbitals(params, res)
            if max(output) > 0.0:
                minComputeDuration = min(minComputeDuration, computeDur)
                minCopyDuration = min(minCopyDuration, copyDur)

        # At some resolutions, on my mobile GPU, I get a totally black image
        # I have no idea why this happens, but this sits here to discard
        # those results, since their timing seems to be somewhat inaccurate.
        if minComputeDuration < 100000000:
            print "{0},{1},{2}".format(res, minComputeDuration, minCopyDuration)

def main():
    # Parse commandline arguments
    parser = OptionParser(usage="%prog [-b/B] [-c] [-p n,l,m]",
                          version="%prog 0.1")
    parser.add_option("-a", "--ati", action="store_true", default=False,
                      dest="promptForDevice", help="prompt for device")
    parser.add_option("-i", "--individual-bench", action="store_true",
                      default=False, dest="onebench", help="run one benchmark")
    parser.add_option("-b", "--bench", action="store_true", default=False,
                      dest="benchmark", help="run short benchmark")
    parser.add_option("-B", "--long-bench", action="store_true",
                      default=False, dest="longBenchmark",
                      help="run long benchmark")
    parser.add_option("-c", "--cpu", action="store_true", default=False,
                      dest="useCPU", help="run on CPU instead of GPU")
    parser.add_option("-p", None, default="3,2,0", dest="params",
                      help="choose parameters to wavefunction")
    parser.add_option("-r", None, default="400", dest="res",
                      help="choose resolution of output image")
    (options, args) = parser.parse_args()

    params = tuple([int(a) for a in options.params.split(",")])

    if not options.promptForDevice:
        if options.useCPU:
            main.ctx = cl.Context(dev_type=cl.device_type.CPU)
        else:
            main.ctx = cl.Context(dev_type=cl.device_type.GPU)
    else:
        main.ctx = cl.create_some_context()

    # Output device(s) being used for computation
    if not (options.benchmark or options.longBenchmark or options.onebench):
        print "Running on:"
        for dev in main.ctx.get_info(cl.context_info.DEVICES):
            print "   ",
            print dev.get_info(cl.device_info.VENDOR),
            print dev.get_info(cl.device_info.NAME)
        print

    # Load and compile the OpenCL kernel
    main.queue = cl.CommandQueue(main.ctx)
    kernelFile = open('hydrogen.cl', 'r')
    main.prg = cl.Program(main.ctx, kernelFile.read()).build()
    kernelFile.close()

    # Evaluate psi with the given parameters, in the given mode
    if options.benchmark:
        benchmark(100, params=params)
    elif options.longBenchmark:
        benchmark(10, params=params)
    elif options.onebench:
        (output, (computeD, copyD)) = renderOrbitals(params, int(options.res))
        print "{0},{1},{2}".format(options.res, computeD, copyD)
    else:
        (output, (computeD, copyD)) = renderOrbitals(params, int(options.res))
        exportImage(output)
        print computeD, copyD

if __name__ == "__main__":
    sys.exit(main())