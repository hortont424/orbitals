float fact(float n)
{
    float f = 1.0;

    if(n <= 0.0)
        return 1.0;

    while (n > 1.0)
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
            return -sqrt((float)1.0 - (x * x));
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
            return -15.0 * pow((float)1.0f - (x * x), (float)(3.0f / 2.0f));
    }
}

float EP(float m)
{
    if(m >= 0)
    {
        return pow(-1.0f, m);
    }
    else
    {
        return 1.0;
    }
}

float2 cnew(float x, float y)
{
    float2 c;
    c.x = x;
    c.y = y;
    return c;
}

float2 cnewf(float x)
{
    float2 c;
    c.x = x;
    c.y = 0.0;
    return c;
}

float2 csqrtf(float a)
{
    if(a <= 0.0)
        return cnew(0.0, sqrt(a * -1.0f));

    return cnewf(sqrt(a));
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

float2 cconj(float2 a)
{
    return cnew(a.x, -a.y);
}

float2 Y(int m, int l, float theta, float phi)
{
    float rootFirst = (2.0 * l + 1.0) / (4 * 3.1415926);
    float rootSecond = fact(l - abs(m)) / fact(l + abs(m));
    float2 root = cmul(cnewf(EP(m)), csqrtf(rootFirst * rootSecond));
    float2 eiStuff = cmul(cexp(cnew(0.0, m * phi)),
                          cnewf(P(m, l, native_cos(theta))));
    return cmul(root, eiStuff);
}

__kernel void density(__global float ipsi,
                      __global int n, __global int l,
                      __global int m, __global float * output,
                      __global int resolution)
{
    int gid = get_global_id(0);

    float theta, phi, r;
    float4 pos = {0, 0, 0, 0};
    int2 imgpos = {0, 0};
    float a = 100.0;
    float2 psi;
    float psiStarPsi;

    // Find coordinates in image from global ID
    imgpos.x = gid % resolution;
    imgpos.y = floor((float)gid / (float)resolution);

    pos.x = ((float)imgpos.x / (float)resolution) - 0.5;
    pos.z = ((imgpos.y % resolution) / (float)resolution) - 0.5;

    pos.x *= 0.35;
    pos.z *= 0.35;

    for(float z = -10.0f; z < 10.0f; z += 0.01)
    {
        // Find coordinates in atomic coordinate space from image coordinates
        pos.y = z;

        // Convert cartesian coordinates to spherical
        r = length(pos);
        theta = acos(pos.z / r);
        phi = atan2(pos.y, pos.x);

        psi = cmul(cmul(cexp(cnewf(-(r / n * a))),
                        cnewf(pow((float)(2.0 * r) / (n * a), (float)l))),
                   cnewf(L(2 * l + 1, n - l - 1, ((2.0 * r) / (n * a)))));
        psi = cmul(cmul(psi, Y(m, l, theta, phi)), cnewf(ipsi));

        psiStarPsi = cmul(cconj(psi), psi).x;

        output[imgpos.x + (imgpos.y * resolution)] += psiStarPsi;
    }
}