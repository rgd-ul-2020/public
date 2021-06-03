#ifndef THESIS_WAV_H
#define THESIS_WAV_H

#include<cstdio>
#include<cstdint>

class Wav
{
    public:
        uint16_t format;
        uint16_t channels;
        uint32_t sample_rate;
        uint16_t bits_per_sample;

        Wav(const char *path, const char *mode);
        virtual ~Wav();

        void read_header();
        void write_header();

        bool eof();
        
        int8_t  read_byte();
        int16_t read_short();
        int32_t read_long();

        uint8_t  read_ubyte();
        uint16_t read_ushort();
        uint32_t read_ulong();

        void write_byte(int8_t data);
        void write_short(int16_t data);
        void write_long(int32_t data);

        void write_ubyte(uint8_t data);
        void write_ushort(uint16_t data);
        void write_ulong(uint32_t data);

    private:
        FILE *file;

        uint32_t byte_rate;
        uint16_t block_align;
        uint32_t data_fseek;
        uint32_t data_size;

        bool update_data;
};

#endif
