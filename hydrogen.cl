typedef float2 Complex;

float fact(float n)
{
    // Factorial... slow?
    float f = 1.0f;

    if(n <= 0.0f)
        return 1.0f;

    while (n > 1.0f)
        f *= n--;

    return f;
}

int L(int a, int b, int x)
{
    // Easy cases
    if(b == 0)
        return fact(a);
    else if(b == 1)
        return fact(a + 1) * (a + 1 - x);

    if(a == 0)
    {
        if(b == 2)
            return 2 - (4 * x) + (x * x);
        else if(b == 3)
            return 6 - (18 * x) + (9 * x * x) - (x * x * x);
        else if(b == 4)
            return 24 - (96 * x) + (72 * x * x) - (16 * x * x * x) + (x * x * x * x);
    }
    else if(a == 1)
    {
        if(b == 2)
            return 3 * (6 - (6 * x) + (x * x));
        else if(b == 3)
            return 4 * (24 - (36 * x) + (12 * x * x) - (x * x * x));
        else if(b == 4)
            return 5 * (120 - (240 * x) + (120 * x * x) - (20 * x * x * x) + (x * x * x * x));
    }
    else if(a == 2)
    {
        if(b == 2)
            return 12 * (12 - (8 * x) + (x * x));
        else if(b == 3)
            return 20 * (60 - (60 * x) + (15 * x * x) - (x * x * x));
        else if(b == 4)
            return 30 * (360 - (480 * x) + (180 * x * x) - (24 * x * x * x) + (x * x * x * x));
    }
    else if(a == 3)
    {
        if(b == 2)
            return 60 * (20 - (10 * x) + (x * x));
        else if(b == 3)
            return 120 * (120 - (90 * x) + (18 * x * x) - (x * x * x));
        else if(b == 4)
            return 210 * (840 - (840 * x) + (252 * x * x) - (28 * x * x * x) + (x * x * x * x));
    }
    else if(a == 4)
    {
        if(b == 2)
            return 360 * (30 - (12 * x) + (x * x));
        else if(b == 3)
            return 840 * (210 - (126 * x) + (21 * x * x) - (x * x * x));
        else if(b == 4)
            return 1680 * (1680 - (1344 * x) + (336 * x * x) - (32 * x * x * x) + (x * x * x * x));
    }

    return 0.0f;
}

float P(int a, int b, float x)
{
    if(a == 0)
    {
        if(b == 0)
            return 1.0f;
        else if(b == 1)
            return x;
        else if(b == 2)
            return 0.5f * (-1 + (3.0f * x * x));
        else if(b == 3)
            return 0.5f * ((-3.0f * x) + (5.0f * x * x * x));
        else if(b == 4)
            return ((3.0f - (30.0f * x * x) + (35.0f * x * x * x * x)) / 8.0f);
    }
    else if(a == 1)
    {
        if(b == 0)
            return 0.0f;
        else if(b == 1)
            return -sqrt((float)1.0f - (x * x));
        else if(b == 2)
            return (-3.0f * x) * sqrt((float)1.0f - (x * x));
        else if(b == 3)
            return - (3.0f / 2.0f) * sqrt((float)1.0f - (x * x)) *
                (-1.0f + (5.0f * x * x));
        else if(b == 4)
            return ((-(5.0f / 2.0f) * sqrt(1.0f - (x * x)) * (-3.0f * x + (7 * x * x * x))));
    }
    else if(a == 2)
    {
        if(b == 0)
            return 0.0f;
        else if(b == 1)
            return 0.0f;
        else if(b == 2)
            return -3.0f * (-1.0f + (x * x));
        else if(b == 3)
            return -(15.0f * x) * (-1.0f + (x * x));
        else if(b == 4)
            return -(15.0f / 2.0f) * ((x * x) - 1.0f) * ((7.0f * x * x) - 1.0f);
    }
    else if(a == 3)
    {
        if(b == 0)
            return 0.0f;
        else if(b == 1)
            return 0.0f;
        else if(b == 2)
            return 0.0f;
        else if(b == 3)
            return -15.0f * pow((float)1.0f - (x * x), (float)(3.0f / 2.0f));
        else if(b == 4)
            return -105.0f * x * pow((float)1.0f - (x * x), 3.0f / 2.0f);
    }
    else if(a == 4)
    {
        if(b == 0)
            return 0.0f;
        else if(b == 1)
            return 0.0f;
        else if(b == 2)
            return 0.0f;
        else if(b == 3)
            return 0.0f;
        else if(b == 4)
            return 105.0f * pow((x * x) - 1.0f, 2.0f);
    }

    return 0.0f;
}

float EP(float m)
{
    if(m >= 0)
        return pow(-1.0f, m);

    return 1.0f;
}

Complex cnew(float x, float y)
{
    // Create a new complex number
    Complex c;
    c.x = x;
    c.y = y;
    return c;
}

Complex cnewf(float x)
{
    // Create a new complex number with only a real component
    Complex c;
    c.x = x;
    c.y = 0.0f;
    return c;
}

Complex csqrtf(float a)
{
    // Take the square root of the complex number a
    if(a <= 0.0f)
        return cnew(0.0f, sqrt(a * -1.0f));

    return cnewf(sqrt(a));
}

Complex cexp(Complex a)
{
    // Compute the exponential e to the complex number a
    float module = exp(a.x);
    float angle = a.y;
    return cnew(module * native_cos(angle), module * native_sin(angle));
}

Complex cmul(Complex a, Complex b)
{
    // Multiply two complex numbers
    return cnew(mad(-a.y, b.y, a.x * b.x), mad(a.y, b.x, a.x * b.y));
}

Complex cconj(Complex a)
{
    // Take the complex conjugate of a
    return cnew(a.x, -a.y);
}

Complex Y(int m, int l, float theta, float phi)
{
    float rootFirst = (2.0f * l + 1.0f) / (4.0f * 3.1415926f);
    float rootSecond = fact(l - abs(m)) / fact(l + abs(m));
    Complex root = cmul(cnewf(EP(m)), csqrtf(rootFirst * rootSecond));
    Complex eiStuff = cmul(cexp(cnew(0.0f, m * phi)),
                          cnewf(P(m, l, native_cos(theta))));
    return cmul(root, eiStuff);
}

__kernel void density(float ipsi,
                      int n, int l,
                      int m, __global float * output,
                      int resolution)
{
    int gid = get_global_id(0);

    float theta, phi, r;
    float4 pos = {0, 0, 0, 0};
    int2 imgpos = {0, 0};
    float a = 100.0f;
    Complex psi;
    float psiStarPsi;

    // Find coordinates in image from global ID
    imgpos.x = gid % resolution;
    imgpos.y = floor((float)gid / (float)resolution);

    // Find coordinates in atomic coordinate space from image coordinates
    pos.x = ((float)imgpos.x / (float)resolution) - 0.5f;
    pos.z = ((imgpos.y % resolution) / (float)resolution) - 0.5f;

    // Arbitrary scale factor, based on parameters
    // TODO: find a way to generate this
    pos.x *= 0.35f;
    pos.z *= 0.35f;

    for(float z = -10.0f; z < 10.0f; z += 0.01f)
    {
        // Choose a y in atomic coordinate space to evaluate at; we iterate
        // through (-10, 10) by 0.01, taking 2000 samples per pixel
        pos.y = z;

        // Convert cartesian coordinates to spherical
        r = length(pos);
        theta = acos(pos.z / r);
        phi = atan2(pos.y, pos.x);

        // Evaluate psi
        psi = cmul(cmul(cexp(cnewf(-(r / n * a))),
                        cnewf(pow((float)(2.0f * r) / (n * a), (float)l))),
                   cnewf(L(2 * l + 1, n - l - 1, ((2.0f * r) / (n * a)))));
        psi = cmul(cmul(psi, Y(m, l, theta, phi)), cnewf(ipsi));

        // Normalize psi
        psiStarPsi = cmul(cconj(psi), psi).x;

        output[imgpos.x + (imgpos.y * resolution)] += psiStarPsi;
    }
}