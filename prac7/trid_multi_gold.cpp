// trid_multi_gold.cpp

void gold_trid_multi(int NX, int M, int niter, float* u, float* c)
{
  float lambda=1.0f, aa, bb, cc, dd;

  // Loop over each of the M independent systems
  for (int m=0; m<M; m++) {
    int offset = m * NX;

    for (int iter=0; iter<niter; iter++) {
      //
      // forward pass
      //
      aa   = -1.0f;
      bb   =  2.0f + lambda;
      cc   = -1.0f;
      dd   = lambda*u[offset + 0];

      bb   = 1.0f / bb;
      cc   = bb*cc;
      dd   = bb*dd;
      c[offset + 0] = cc;
      u[offset + 0] = dd;

      for (int i=1; i<NX; i++) {
        aa   = -1.0f;
        bb   = 2.0f + lambda - aa*c[offset+i-1];
        dd   = lambda*u[offset+i] - aa*u[offset+i-1];
        bb   = 1.0f/bb;
        cc   = -bb;
        dd   = bb*dd;
        c[offset + i] = cc;
        u[offset + i] = dd;
      }

      //
      // reverse pass
      //
      u[offset + NX-1] = dd;

      for (int i=NX-2; i>=0; i--) {
        dd   = u[offset+i] - c[offset+i]*dd;
        u[offset+i] = dd;
      }
    }
  }
}