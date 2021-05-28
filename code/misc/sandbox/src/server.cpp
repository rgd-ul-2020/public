#include<iostream>
#include<unistd.h>
#include<cstdlib>

int main(int argc, char **argv)
{
    unsigned char buffer[65536];
    uint16_t buffer_size;

    while ((buffer_size = read(0, &buffer, 65535)) > 0) {
        buffer[buffer_size] = 0;
        write(1, buffer, buffer_size);
    }

    return EXIT_SUCCESS;
}
