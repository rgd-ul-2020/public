#include<iostream>

#include<cstdlib>
#include<cstring>

#include<unistd.h>
#include<sndfile.h>

#include "wav.hpp"

#define BUFFER_SIZE 5

uint8_t encode(int16_t sample)
{
    return (uint8_t) ((sample / 256) + 128);
}

int main(int argc, const char **argv)
{
    time_t start = time(NULL);

    if (argc < 2) {
        std::cerr << "Please provide a file name." << std::endl;
        return EXIT_FAILURE;
    }
    
    Wav file(argv[1], "r");
    file.read_header();

    uint8_t next;

    while (!file.eof()) {
        next = encode(file.read_short());
        write(1, &next, 1);
    }

    std::cerr << "Encoding: " << time(NULL) - start << "s" << std::endl;
    
    return EXIT_SUCCESS;
}
