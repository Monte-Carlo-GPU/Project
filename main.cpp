#include <stdio.h>
#include <vector>
#include <time.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <curand.h>
#include <limits>     
#include <cfloat>     
#include <cmath>      

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440  // Value 1/sqrt(2)
#endif

#include "kernel.cuh"

using namespace std;

// CUDA and cuRAND error checking macros
#define CUDA_CHECK(call) do {                                \
    cudaError_t err = call;                                  \
    if (err != cudaSuccess) {                                \
        cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " " \
             << cudaGetErrorString(err) << endl;             \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

#define CURAND_CHECK(call) do {                              \
    curandStatus_t status = call;                            \
    if (status != CURAND_STATUS_SUCCESS) {                   \
        cerr << "cuRAND error at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// Safe input function
template<typename T>
bool safeInput(T& val) {
    cin >> val;
    if (!cin.good()) {
        cin.clear();
        cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        return false;
    }
    return true;
}

// Normal CDF using error function
double normCDF(double x) {
    return 0.5 * erfc(-x * M_SQRT1_2);
}

// Black-Scholes formula for a European call
double blackScholesCall(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S/K) + (r + 0.5 * sigma * sigma)*T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return S * normCDF(d1) - K * exp(-r*T)*normCDF(d2);
}

// Black-Scholes formula for a European put
double blackScholesPut(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S/K) + (r + 0.5 * sigma * sigma)*T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return K * exp(-r*T)*normCDF(-d2) - S*normCDF(-d1);
}

int main() {
    int choix;
    size_t N_PATHS = 100000;
    size_t N_STEPS = 365;
    float T = 1.0f;
    float K = 100.0f;
    float S0 = 100.0f;
    float sigma = 0.2f;
    float mu = 0.1f;
    float r = 0.01f;

    cout << "Choose the type of option you want to price:" << endl << endl;
    cout << "1. Call Option." << endl;
    cout << "2. Put Option." << endl << endl;
    cout << "Your choice (1-2): ";

    if (!safeInput(choix) || (choix != 1 && choix != 2)) {
        cerr << "Error: Invalid choice." << endl;
        return EXIT_FAILURE;
    }

    if (choix == 1) {
        cout << "You have chosen a Call Option." << endl << endl;
    } else {
        cout << "You have chosen a Put Option." << endl << endl;
    }

    cout << "Number of trajectories to simulate (default 100000): "; 
    if (!safeInput(N_PATHS) || N_PATHS == 0) {
        cerr << "Error: N_PATHS must be a positive number." << endl;
        return EXIT_FAILURE;
    }

    cout << "Number of steps per trajectory (default 365): "; 
    if (!safeInput(N_STEPS) || N_STEPS == 0) {
        cerr << "Error: N_STEPS must be a positive number." << endl;
        return EXIT_FAILURE;
    }

    cout << "Underlying asset price (default 100): "; 
    if (!safeInput(S0) || S0 <= 0.0f) {
        cerr << "Error: S0 must be > 0." << endl;
        return EXIT_FAILURE;
    }

    cout << "Strike price (default 100): "; 
    if (!safeInput(K) || K < 0.0f) {
        cerr << "Error: K must be >= 0." << endl;
        return EXIT_FAILURE;
    }

    cout << "Risk-free rate (default 0.01): "; 
    if(!safeInput(r)) {
        cerr << "Error: Invalid input for r." << endl;
        return EXIT_FAILURE;
    }

    cout << "Volatility (default 0.2): "; 
    if (!safeInput(sigma) || sigma < 0.0f) {
        cerr << "Error: sigma must be >= 0." << endl;
        return EXIT_FAILURE;
    }


    cout << "Time to maturity in years (default 1): ";
    if (!safeInput(T) || T <= 0.0f) {
        cerr << "Error: T must be > 0." << endl;
        return EXIT_FAILURE;
    }

    cout << "Annual drift (default 0.1): "; 
    if(!safeInput(mu)) {
        cerr << "Error: Invalid input for mu." << endl;
        return EXIT_FAILURE;
    }
    cout << endl;

    // Get GPU memory info
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    cout << "Available GPU memory: " << free_mem / (1024 * 1024) << " MB" << endl;
    cout << "Total GPU memory: " << total_mem / (1024 * 1024) << " MB" << endl;

    // Compute memory per trajectory
    if (N_STEPS > (std::numeric_limits<size_t>::max() - 1)) {
        cerr << "Error: N_STEPS too large, causing overflow." << endl;
        return EXIT_FAILURE;
    }

    size_t steps_plus_one = N_STEPS + 1;
    if (steps_plus_one > (std::numeric_limits<size_t>::max() / sizeof(float))) {
        cerr << "Error: Overflow in calculating memory per trajectory." << endl;
        return EXIT_FAILURE;
    }

    size_t bytes_per_trajectory = steps_plus_one * sizeof(float);

    double safe_factor = 0.8;
    double safe_memory = free_mem * safe_factor;
    if (bytes_per_trajectory == 0) {
        cerr << "Error: bytes_per_trajectory is zero, unexpected." << endl;
        return EXIT_FAILURE;
    }

    size_t max_batch_size = static_cast<size_t>(safe_memory / bytes_per_trajectory);
    if (max_batch_size == 0) {
        cerr << "Error: Not enough GPU memory even for one trajectory. Try reducing N_STEPS." << endl;
        return EXIT_FAILURE;
    }

    size_t batch_size = max_batch_size;

    unsigned long long N_PATHS_ull = static_cast<unsigned long long>(N_PATHS);
    unsigned long long batch_size_ull = static_cast<unsigned long long>(batch_size);
    unsigned long long numerator = N_PATHS_ull + batch_size_ull - 1ULL;
    if (batch_size_ull == 0ULL) {
        cerr << "Error: batch_size_ull is zero." << endl;
        return EXIT_FAILURE;
    }

    unsigned long long total_batches_ull = numerator / batch_size_ull;
    if (total_batches_ull > std::numeric_limits<size_t>::max()) {
        cerr << "Error: total_batches too large, cannot fit in size_t." << endl;
        return EXIT_FAILURE;
    }

    size_t total_batches = static_cast<size_t>(total_batches_ull);

    if (N_PATHS > batch_size) {
        cout << "Processing " << N_PATHS << " trajectories in " << total_batches 
             << " batches of up to " << batch_size << " trajectories each." << endl;
    } else {
        cout << "Processing all " << N_PATHS << " trajectories in a single batch." << endl;
    }

    float *d_S_batch, *d_normals_batch;
    CUDA_CHECK(cudaMalloc((void**)&d_S_batch, batch_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_normals_batch, batch_size * N_STEPS * sizeof(float)));

    curandGenerator_t curandGenerator;
    CURAND_CHECK(curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    vector<float> h_S(batch_size);

    double price_GPU = 0.0;
    clock_t start_time = clock();

// Generate normal numbers and run kernel on GPU for all trajectories
// Assume all N_PATHS <= batch_size for simplicity (or you have already chosen such parameters)
// If N_PATHS > batch_size, batch processing will occur, but one batch is assumed for simplicity
    size_t current_batch_size = N_PATHS; 
    size_t current_N_NORMALS = current_batch_size * N_STEPS;

    CURAND_CHECK(curandGenerateNormal(curandGenerator, d_normals_batch, current_N_NORMALS, 0.0f, sqrt((float)T/(float)N_STEPS)));


    float dt = (float)T/(float)N_STEPS;
    if (choix == 1) {
        mc_call_GPU(d_S_batch, T, K, S0, sigma, mu, r, dt, d_normals_batch, (unsigned)N_STEPS, (unsigned)current_batch_size);
    } else {
        mc_put_GPU(d_S_batch, T, K, S0, sigma, mu, r, dt, d_normals_batch, (unsigned)N_STEPS, (unsigned)current_batch_size);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(&h_S[0], d_S_batch, current_batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (size_t i = 0; i < current_batch_size; i++) {
        price_GPU += h_S[i];
    }

    clock_t end_time = clock();
    double elapsed_time = double(end_time - start_time) / CLOCKS_PER_SEC * 1000.0; // ms

    price_GPU /= N_PATHS;

    if (!std::isfinite(price_GPU)) {
        cerr << "Error: Calculated GPU price is not a finite number (NaN or Inf)." << endl;
        CUDA_CHECK(cudaFree(d_S_batch));
        CUDA_CHECK(cudaFree(d_normals_batch));
        CURAND_CHECK(curandDestroyGenerator(curandGenerator));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return EXIT_FAILURE;
    }

    // Black-Scholes Analytical Price
    double analytical_price = 0.0;
    if (choix == 1) {
        analytical_price = blackScholesCall(S0, K, r, sigma, T);
    } else {
        analytical_price = blackScholesPut(S0, K, r, sigma, T);
    }

    // Логирование в results.txt
    {
    std::ofstream outfile("results.txt", std::ios::app);
        if (outfile.is_open()) {
            outfile << "Option Type: " << (choix == 1 ? "Call" : "Put") << "\n";
            outfile << "Number of trajectories: " << N_PATHS << "\n";
            outfile << "Number of steps: " << N_STEPS << "\n";
            outfile << "Underlying asset price: " << S0 << "\n";
            outfile << "Strike price: " << K << "\n";
            outfile << "Risk-free rate: " << r << "\n";
            outfile << "Volatility: " << sigma << "\n";
            outfile << "Time to maturity (years): " << T << "\n";
            outfile << "Annual drift: " << mu << "\n";
            outfile << "Option price (GPU Monte Carlo): " << price_GPU << "\n";
            outfile << "Option price (Black-Scholes): " << analytical_price << "\n";
            outfile << "Difference: " << (price_GPU - analytical_price) << "\n";
            outfile << "Monte Carlo Computation on GPU: " << elapsed_time << " ms\n";
            outfile << "-----------------------------------------------\n";
            outfile.close();
        } else {
            cerr << "Error: Could not open results.txt for writing (append mode)." << endl;
        }
    }


    

    // Print to screen
    cout << "********************* INFO *********************" << endl;
    cout << "Number of trajectories: " << N_PATHS << endl;
    cout << "Number of steps: " << N_STEPS << endl;
    cout << "Underlying asset price: " << S0 << endl;
    cout << "Strike price: " << K << endl;
    cout << "Risk-free rate: " << r << endl;
    cout << "Volatility: " << sigma << endl;
    cout << "Time to maturity (years): " << T << endl;
    cout << "Annual drift: " << mu << endl;
    cout << "********************* PRICE *********************" << endl;
    cout << "Option price (GPU Monte Carlo): " << price_GPU << endl;
    cout << "Option price (Black-Scholes): " << analytical_price << endl;
    cout << "Difference: " << (price_GPU - analytical_price) << endl;
    cout << "********************* EXECUTION TIME **************" << endl;
    cout << "Monte Carlo Computation on GPU: " << elapsed_time << " ms" << endl;

    // Saving multiple trajectories to trajectories.csv
    size_t N_TRAJ_TO_SAVE = ( N_PATHS > 100 ) ? N_PATHS : 100;
    if (N_PATHS >= N_TRAJ_TO_SAVE) {
        // Let's copy the normal numbers for the trajectories
        std::vector<float> h_normals_for_save(N_TRAJ_TO_SAVE * N_STEPS);

        CUDA_CHECK(cudaMemcpy(h_normals_for_save.data(), d_normals_batch,
                              N_TRAJ_TO_SAVE * N_STEPS * sizeof(float),
                              cudaMemcpyDeviceToHost));


        double dt_ = double(T) / double(N_STEPS);
        double drift_ = (r - 0.5 * sigma * sigma) * dt_;
        double vol_ = sigma * sqrt(dt_);

        std::vector<std::vector<double>> trajectories(N_TRAJ_TO_SAVE, std::vector<double>(N_STEPS+1, S0));

        for (size_t i = 0; i < N_TRAJ_TO_SAVE; i++) {
            double S_curr = S0;
            for (size_t step = 1; step <= N_STEPS; step++) {
                double Z = h_normals_for_save[i*N_STEPS + (step-1)];
                S_curr = S_curr * exp(drift_ + vol_ * Z);
                trajectories[i][step] = S_curr;
            }
        }

        std::ofstream trajFile("trajectories.csv");
        if (!trajFile.is_open()) {
            std::cerr << "Error: Could not open trajectories.csv for writing.\n";
        } else {
            trajFile << "time";
            for (size_t i = 0; i < N_TRAJ_TO_SAVE; i++) {
                trajFile << ",traj_" << i;
            }
            trajFile << "\n";

            for (size_t step = 0; step <= N_STEPS; step++) {
                double current_time = step * dt_;
                trajFile << current_time;
                for (size_t i = 0; i < N_TRAJ_TO_SAVE; i++) {
                    trajFile << "," << trajectories[i][step];
                }
                trajFile << "\n";
            }
            trajFile.close();
            std::cout << "Trajectories saved to trajectories.csv\n";
        }
    }

    CUDA_CHECK(cudaFree(d_S_batch));
    CUDA_CHECK(cudaFree(d_normals_batch));
    CURAND_CHECK(curandDestroyGenerator(curandGenerator));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}


