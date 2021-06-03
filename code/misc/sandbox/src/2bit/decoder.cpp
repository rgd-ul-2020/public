#include<iostream>

#include<cstdlib>
#include<cstring>

#include<unistd.h>
#include<sndfile.h>

#include "wav.hpp"

#define BUFFER_SIZE 5

// ref. https://github.com/erikd/libsndfile
static short b2_decode[4] = { 
    5000, 10000, -10000, -5000
}; 

int16_t decode(uint8_t sample, int pos)
{
    return b2_decode[(sample >> (2 * (3 - pos))) & 0x03];
}

int main(int argc, const char **argv)
{
    time_t start = time(NULL);

    if (argc < 2) {
        std::cerr << "Please provide a file name." << std::endl;
        return EXIT_FAILURE;
    }
    
    Wav file(argv[1], "w");
    file.bits_per_sample = 16;
    file.write_header();

    uint8_t next;

    while (read(0, &next, 1) > 0) {
        for (int i = 0; i < 4; i++) {
            file.write_short(decode(next, i));
        }
    }

    std::cerr << "Decoding: " << time(NULL) - start << "s" << std::endl;
    
    return EXIT_SUCCESS;
}

