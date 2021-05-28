#include<iostream>

#include<cstdlib>
#include<cstring>

#include<unistd.h>
#include<sndfile.h>

#include "wav.hpp"

#define BUFFER_SIZE 5

int16_t decode(uint8_t sample)
{
    return (sample - 128) * 256;
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
        file.write_short(decode(next));
    }

    std::cerr << "Decoding: " << time(NULL) - start << "s" << std::endl;
    
    return EXIT_SUCCESS;
}

