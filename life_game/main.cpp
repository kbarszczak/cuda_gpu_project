#include <iostream>

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

    void requireWithinBound(int x, int y) {
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

    const State &get(int x, int y) {
        requireWithinBound(x, y);
        return states[y][x];
    }

    [[nodiscard]] int getWidth() const {
        return width;
    }

    [[nodiscard]] int getHeight() const {
        return height;
    }

};

int main() {
    cout << "Hello, World!" << endl;
    return 0;
}
