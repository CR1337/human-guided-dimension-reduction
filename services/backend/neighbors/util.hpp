#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <iostream>

static void print(const char *message) {
    std::cout << message << '\n';
}

static void printError(const char *message) {
    std::cerr << message << '\n';
}

#endif // __UTIL_HPP__
