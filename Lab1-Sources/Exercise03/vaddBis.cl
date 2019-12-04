//------------------------------------------------------------------------------
//
// kernel:  vadd
//
// Purpose: Compute the elementwise sum c = a+b
//
// input: a and b float vectors of length count
//
// output: c float vector of length count holding the sum a + b
//

__kernel void vaddBis(
   __global float* a,
   __global float* b,
   __global float* c,
   __global float* d,
   __global float* e,
   __global float* f,
   __global float* g,
   const unsigned int count)
{
   int i = get_global_id(0);
  //   float cLoc;
  //   float dLoc;
   if(i < count)  {
      //  cLoc = a[i] + b[i];
      //  dLoc = cLoc + e[i];
      //  f[i] = dLoc + g[i];
      //  c[i] = cLoc;
      //  d[i] = dLoc;

      c[i] = a[i] + b[i];
      d[i] = c[i] + e[i];
      f[i] = d[i] + g[i];
   }
}
