#ifndef KERNEL_CUH
#define KERNEL_CUH

void mc_call_GPU(float * d_s, float T, float K, float S0, float sigma, float mu, float r, float dt, float * d_normals, unsigned N_STEPS, unsigned N_PATHS);
void mc_put_GPU(float * d_s, float T, float K, float S0, float sigma, float mu, float r, float dt, float * d_normals, unsigned N_STEPS, unsigned N_PATHS);

#endif

