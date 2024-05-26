#include <iostream>
#include <cuda_runtime.h>

using namespace std;

enum State {
    Alive, Dead
};

class Board {

private:

    const int width, height;
    State **states; // states[height][width]

    void acquireMem(int _width, int _height) {
        states = new State *[_width * _height];
        states[0] = new State[_width * _height];
        for (int i = 0; i < _height; ++i) {
            states[i] = states[0] + i * _width;
        }
    }

    void copy(const Board &other) {
        acquireMem(other.width, other.height);
        for (int i = 0; i < other.height; ++i) {
            for (int j = 0; j < other.width; ++j) {
                states[i][j] = other.states[i][j];
            }
        }
    }

    void move(Board &other) {
        this->states = other.states;
        other.states = nullptr;
    }

    void free() {
        delete[] states[0];
        delete[] states;
    }

    void requireWithinBound(int x, int y) const {
        if (x < 0 | x >= width) throw invalid_argument("x it out of bound");
        if (y < 0 | y >= width) throw invalid_argument("y it out of bound");
    }

public:

    Board(int width, int height) : width(width), height(height), states(new State *[width * height]) {
        acquireMem(width, height);
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                states[i][j] = State::Dead;
            }
        }
    }

    Board(const Board &other) : width(other.width), height(other.height), states(nullptr) {
        copy(other);
    }

    Board &operator=(const Board &other) {
        if (this != &other) {
            copy(other);
        }
        return *this;
    }

    Board(Board &&other) noexcept: width(other.width), height(other.height), states(nullptr) {
        move(other);
    }

    ~Board() {
        free();
    }

    void set(State state, int x, int y) {
        requireWithinBound(x, y);
        states[y][x] = state;
    }

    [[nodiscard]] const State &get(int x, int y) const {
        requireWithinBound(x, y);
        return states[y][x];
    }

    [[nodiscard]] int getWidth() const {
        return width;
    }

    [[nodiscard]] int getHeight() const {
        return height;
    }

    friend std::ostream &operator<<(std::ostream &out, const Board &board) {
        for (int i = -1; i <= board.height; ++i) {
            for (int j = -1; j <= board.width; ++j) {
                if (j < 0 || j >= board.width) out << "|";
                else if (i < 0 || i >= board.height) out << "-";
                else {
                    switch (board.states[i][j]) {
                        case Alive:
                            out << "x";
                            break;
                        case Dead:
                            out << " ";
                            break;
                    }
                }
            }
            out << "\n";
        }
        return out;
    }

};

class StateProcessor {

public:

    [[nodiscard]] virtual Board *next(const Board *board) const = 0;

};

class CPUStateProcessor : public StateProcessor {

private:

    [[nodiscard]] static int countNeighbours(const Board *board, int x, int y) {
        int result = 0;
        for (int i = -1; i < 2; ++i) {
            for (int j = -1; j < 2; ++j) {
                if (i == 0 && j == 0) continue;

                int _x = x + j, _y = y + i;
                if (_x < 0 || _y < 0 || _x >= board->getWidth() || _y >= board->getHeight()) continue;

                if (board->get(_x, _y) == State::Alive) ++result;
            }
        }
        return result;
    }

public:

    [[nodiscard]] Board *next(const Board *board) const override {
        auto *result = new Board(*board);
        for (int i = 0; i < board->getHeight(); ++i) {
            for (int j = 0; j < board->getWidth(); ++j) {
                int neighbours = countNeighbours(board, j, i);

                switch (board->get(j, i)) {
                    case Alive:
                        if (neighbours <= 1 || neighbours >= 4) result->set(State::Dead, j, i);
                        break;
                    case Dead:
                        if (neighbours == 3) result->set(State::Alive, j, i);
                        break;
                }
            }
        }

        return result;
    }

};

class GPUStateProcessor : public StateProcessor {

private:

    static __device__ int countNeighbours(const Board *board, int x, int y) {
        int result = 0;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                if (i == 0 && j == 0) continue;

                int _x = x + j, _y = y + i;
                if (_x < 0 || _y < 0 || _x >= board->getWidth() || _y >= board->getHeight()) continue;

                if (board->get(_x, _y) == State::Alive) ++result;
            }
        }
        return result;
    }

    static __global__ void processBoard(const Board *input, Board *output) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < input->getWidth() && y < input->getHeight()) {
            int neighbours = countNeighbours(input, x, y);

            switch (input->get(x, y)) {
                case State::Alive:
                    if (neighbours <= 1 || neighbours >= 4) {
                        output->set(State::Dead, x, y);
                    } else {
                        output->set(State::Alive, x, y);
                    }
                    break;
                case State::Dead:
                    if (neighbours == 3) {
                        output->set(State::Alive, x, y);
                    } else {
                        output->set(State::Dead, x, y);
                    }
                    break;
            }
        }
    }

public:

    [[nodiscard]] Board *next(const Board *board) const override {
        Board *d_input, *d_output;

        size_t size = board->getWidth() * board->getHeight() * sizeof(State);

        // Allocate device memory
        cudaMalloc(&d_input, sizeof(Board));
        cudaMalloc(&d_output, sizeof(Board));

        // Copy board to device memory
        cudaMemcpy(d_input, board, sizeof(Board), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output, board, sizeof(Board), cudaMemcpyHostToDevice);

        // Set up grid and block dimensions
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((board->getWidth() + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (board->getHeight() + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Launch kernel
        processBoard<<<numBlocks, threadsPerBlock>>>(d_input, d_output);

        // Allocate result board on host
        Board *result = new Board(*board);

        // Copy result from device to host
        cudaMemcpy(result, d_output, sizeof(Board), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);

        return result;
    }

};

int main() {

    string buffer;
    int gameLength = 1000;
    auto *initialBoard = new Board(10, 10);
    auto *processor = new CPUStateProcessor();

    // construct board
    while (true) {
        cout << "Do you want to add alive state? y|n: ";
        cin >> buffer;

        if (buffer == "y") {
            int x, y;
            cout << "x: ";
            cin >> x;
            cout << "y: ";
            cin >> y;

            try {
                initialBoard->set(State::Alive, x, y);
            } catch (const invalid_argument &exception) {
                cerr << "Cannot set alive state because of [" << exception.what() << "]\n";
            }
        } else {
            cout << "Starting the game ...\n";
            break;
        }
    }

    // main game loop
    Board *previousState = new Board(*initialBoard), *currentState;

    for (int i = 0; i < gameLength; ++i) {
        currentState = processor->next(previousState);
        cout << *currentState;

        delete previousState;
        previousState = currentState;
    }

    delete previousState;
    delete initialBoard;
    delete processor;

    return 0;
}
