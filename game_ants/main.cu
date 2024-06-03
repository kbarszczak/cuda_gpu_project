#include <iostream>
#include <vector>
#include <string>

#define GRID_SIZE  128  // (GRID_SIZE x GRID_SIZE)
#define ANTS_AMOUNT 7
#define ITERATIONS 10000

// Structure representing our ant
struct Ant {
    int x, y;  // Ants position
    int dir;   // Direction (0 = right, 1 = down, 2 = left, 3 = up)
};

// Ant route directions
__device__ int dx[4] = {1, 0, -1, 0};
__device__ int dy[4] = {0, 1, 0, -1};

// Kernel function
__global__ void updateGrid(int* grid, Ant* ant, int gridSize) {
    int i = threadIdx.x;
    int x = ant[i].x;
    int y = ant[i].y;
    int dir = ant[i].dir;

    int idx = y * gridSize + x;
    int state = grid[idx];

    // Update cell state and ant direction
    if (state == 0) {
        ant[i].dir = (dir + 1) % 4;  // Turn right
        grid[idx] = i + 1;           // Flip to color
    } else {
        ant[i].dir = (dir + 3) % 4;  // Turn left
        grid[idx] = 0;               // Flip to white
    }

    // Update ant position
    ant[i].x = (x + dx[ant[i].dir] + gridSize) % gridSize;
    ant[i].y = (y + dy[ant[i].dir] + gridSize) % gridSize;
}

int main() {
    int gridSize = GRID_SIZE;
    int antsAmount = ANTS_AMOUNT;
    size_t gridMemSize = gridSize * gridSize * sizeof(int);
    size_t antsMemSize = antsAmount * sizeof(Ant);

    // Host memory allocation
    int* h_grid = new int[gridSize * gridSize]();
    Ant* h_ants = new Ant[antsAmount]();
    for(int i = 0; i < antsAmount;i++)
    {
        int coordinate = (gridSize / antsAmount) * (i + 1);
        h_ants[i] = {coordinate, coordinate, 0};
    }

    // Device memory allocation
    int* d_grid;
    Ant* d_ants;
    cudaMalloc(&d_grid, gridMemSize);
    cudaMalloc(&d_ants, antsMemSize);

    // Copy data from host to device
    cudaMemcpy(d_grid, h_grid, gridMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ants, h_ants, antsMemSize, cudaMemcpyHostToDevice);

    // Number of iterations
    int iterations = ITERATIONS;

    // Kernel execution
    for (int i = 0; i < iterations; ++i) {
        updateGrid<<<1, antsAmount>>>(d_grid, d_ants, gridSize);
        cudaDeviceSynchronize();
    }

    // Copy results back to host
    cudaMemcpy(h_grid, d_grid, gridMemSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ants, d_ants, antsMemSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_grid);
    cudaFree(d_ants);

    // Display the final grid state
    for (int y = 0; y < gridSize; ++y) {
        for (int x = 0; x < gridSize; ++x) {
            // Create color text by this command
            std::string antPoint = "\033[1;" + std::to_string(31 + (h_grid[y * gridSize + x])%7) + "m#\033[0m";
            std::cout << (h_grid[y * gridSize + x] ? antPoint : ".");
        }
        std::cout << std::endl;
    }

    // Free host memory
    delete[] h_grid;
    delete[] h_ants;

    return 0;
}