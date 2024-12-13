#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "kernel.cuh"

__global__ void mc_kernel_call(float * d_s, float T, float K, float S0, float sigma, float mu, float r, float dt, float * d_normals, unsigned N_STEPS, unsigned N_PATHS)
{
    const unsigned tid = threadIdx.x; 
    const unsigned bid = blockIdx.x; 
    const unsigned bsz = blockDim.x; 
    int s_idx = tid + bid * bsz;
    int n_idx = s_idx;
    float s_curr = S0;

    if (s_idx < N_PATHS) {
        for (unsigned n = 0; n < N_STEPS; n++) {
            s_curr = s_curr + mu * s_curr * dt + sigma * s_curr * d_normals[n_idx];
            n_idx++;
        }
        double payoff = (s_curr > K ? s_curr - K : 0.0);
        d_s[s_idx] = exp(-r*T)*payoff;
    }
}

__global__ void mc_kernel_put(float * d_s, float T, float K, float S0, float sigma, float mu, float r, float dt, float * d_normals, unsigned N_STEPS, unsigned N_PATHS)
{
    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;
    const unsigned bsz = blockDim.x;
    int s_idx = tid + bid * bsz;
    int n_idx = s_idx;
    float s_curr = S0;

    if (s_idx < N_PATHS) {
        for (unsigned n = 0; n < N_STEPS; n++) {
            s_curr = s_curr + mu*s_curr*dt + sigma*s_curr*d_normals[n_idx];
            n_idx++;
        }
        double payoff = (s_curr < K ? K - s_curr : 0.0);
        d_s[s_idx] = exp(-r*T)*payoff;
    }
}

void mc_call_GPU(float * d_s, float T, float K, float S0, float sigma, float mu, float r, float dt, float * d_normals, unsigned N_STEPS, unsigned N_PATHS) 
{
    const unsigned BLOCK_SIZE = 1024; 
    const unsigned GRID_SIZE = (N_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mc_kernel_call<<<GRID_SIZE, BLOCK_SIZE>>>(d_s, T, K, S0, sigma, mu, r, dt, d_normals, N_STEPS, N_PATHS);
}

void mc_put_GPU(float * d_s, float T, float K, float S0, float sigma, float mu, float r, float dt, float * d_normals, unsigned N_STEPS, unsigned N_PATHS)
{
    const unsigned BLOCK_SIZE = 1024;
    const unsigned GRID_SIZE = (N_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mc_kernel_put<<<GRID_SIZE, BLOCK_SIZE>>>(d_s, T, K, S0, sigma, mu, r, dt, d_normals, N_STEPS, N_PATHS);
}
