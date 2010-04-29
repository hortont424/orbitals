Orbitals
========

This project aims to simulate the orbital structure of a hydrogen atom. It
evaluates the quantum wavefunction at a large number of points, composing an
image of the spatial probability of the single electron's location. OpenCL is
used to parallelize the process, resulting in significant speedups.

This software was designed and developed as my final project for RPI's
CSCI-4320, Parallel Programming.

All code is being released under the two-clause BSD license, which can be found
in the toplevel LICENSE file.

Dependencies
------------

* Python
* PyOpenCL
* PIL
* numpy

Benchmarks
----------

Very roughly: right now I'm seeing approximately an 18x speedup using a
NVIDIA GeForce 330M GT GPU over a Intel Core i7 @ 2x2.66/3.33GHz. This leads
me to expect something on the order of 40-50x when I get PyOpenCL working on
the machine which my ATI Radeon 4890 is attached to...