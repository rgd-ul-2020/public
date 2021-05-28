#include<iostream>

#include<cstdlib>
#include<cstring>

#include<unistd.h>
#include<sndfile.h>

#include "wav.hpp"

#define BUFFER_SIZE 5

uint8_t encode(int16_t sample)
{
    if (sample <= 0) {
        return 0;
    } else {
        return 1;
    }
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
    int encoded = 0;

    while (!file.eof()) {
        next = encode(file.read_short()) << (7 - encoded);
        encoded++;

        if (encoded % 8 == 0) {
            write(1, &next, 1);
            encoded = 0;
        }
    }

    if (encoded % 8 != 0) {
        write(1, &next, 1);
    }

    std::cerr << "Encoding: " << time(NULL) - start << "s" << std::endl;
    
    return EXIT_SUCCESS;
}

