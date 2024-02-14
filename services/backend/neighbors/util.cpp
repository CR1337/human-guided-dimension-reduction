#include "util.hpp"

#include <iostream>

void print(const char *message) {
    std::cout << message << '\n';
}

void printError(const char *message) {
    std::cerr << message << '\n';
}
