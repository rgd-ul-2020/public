#include "wav.hpp"

#include<iostream>
#include<exception>

#include<cstring>

Wav::Wav(const char *path, const char *mode)
{
    file = fopen(path, mode);

    format          = 1;
    channels        = 2;
    sample_rate     = 44100;
    bits_per_sample = 8;

    update_data = false;
}

Wav::~Wav()
{
    if (update_data) {
        fseek(file, 4, SEEK_SET);
        write_long(40 + data_size);

        fseek(file, data_fseek + 4, SEEK_SET);
        write_long(data_size);
    }

    fclose(file);
}

void Wav::read_header()
{
    const char *riff = "RIFF";

    for (int i = 0; i < 4; i++) {
        if (read_byte() != riff[i]) {
            throw std::runtime_error("Invalid file");
        }
    }

    const uint32_t chunk_size = read_ulong();

    const char *wave = "WAVE";

    for (int i = 0; i < 4; i++) {
        if (read_byte() != wave[i]) {
            throw std::runtime_error("Invalid format");
        }
    }

    const char *fmt = "fmt ";

    for (int i = 0; i < 4; i++) {
        if (read_byte() != fmt[i]) {
            throw std::runtime_error("No \"fmt \" chunk");
        }
    }

    const uint32_t schunk1_size = read_ulong();
    
    format      = read_ushort();
    channels    = read_ushort();
    sample_rate = read_ulong();
    byte_rate   = read_ulong();
    block_align = read_ushort();

    bits_per_sample = read_ushort();

    for (int i = schunk1_size - 16; i > 0; i -= 8) {
        read_ubyte();
    }
    
    data_fseek = 16 + schunk1_size;

    char     schunk_id[5] = { 0, 0, 0, 0, 0 };
    uint32_t schunk_size = 0;

    while (schunk_id[0] == 0) {
        for (int i = 0; i < 4; i++) {
            schunk_id[i] = read_byte();
        }

        data_fseek += schunk_size;
        schunk_size = read_ulong();

        if (strcmp(schunk_id, "data") == 0) {
            data_size = schunk_size;
            break;

        } else {
            for (int i = 0; i < schunk_size; i++) {
                read_ubyte();
            }
        }

        schunk_id[0] = 0;
    }
}

void Wav::write_header()
{
    byte_rate   = sample_rate * channels * (bits_per_sample / 8);
    block_align = channels * (bits_per_sample / 8);
    
    fwrite("RIFF", 4, 1, file);

    // Chunk Size
    write_ulong(40);

    fwrite("WAVE", 4, 1, file);
    fwrite("fmt ", 4, 1, file);

    // SubChunk Size
    write_ulong(16);

    write_ushort(format);
    write_ushort(channels);
    write_ulong(sample_rate);
    write_ulong(byte_rate);
    write_ushort(block_align);
    write_ushort(bits_per_sample);

    fwrite("data", 4, 1, file);

    data_fseek = 40;

    // Data Size
    write_ulong(0);
}

bool Wav::eof()
{
    return feof(file);
}

int8_t Wav::read_byte()
{
    int8_t data;
    fread(&data, sizeof(data), 1, file);
    return data;
}

int16_t Wav::read_short()
{
    int16_t data;
    fread(&data, sizeof(data), 1, file);
    return data;
}

int32_t Wav::read_long()
{
    int32_t data;
    fread(&data, sizeof(data), 1, file);
    return data;
}

uint8_t Wav::read_ubyte()
{
    uint8_t data;
    fread(&data, sizeof(data), 1, file);
    return data;
}

uint16_t Wav::read_ushort()
{
    uint16_t data;
    fread(&data, sizeof(data), 1, file);
    return data;
}

uint32_t Wav::read_ulong()
{
    uint32_t data;
    fread(&data, sizeof(data), 1, file);
    return data;
}

void Wav::write_byte(int8_t data)
{
    fwrite(&data, sizeof(data), 1, file);
    data_size  += 1;
    update_data = true;
}

void Wav::write_short(int16_t data)
{
    fwrite(&data, sizeof(data), 1, file);
    data_size  += 2;
    update_data = true;
}

void Wav::write_long(int32_t data)
{
    fwrite(&data, sizeof(data), 1, file);
    data_size  += 4;
    update_data = true;
}

void Wav::write_ubyte(uint8_t data)
{
    fwrite(&data, sizeof(data), 1, file);
    data_size  += 1;
    update_data = true;
}

void Wav::write_ushort(uint16_t data)
{
    fwrite(&data, sizeof(data), 1, file);
    data_size  += 2;
    update_data = true;
}

void Wav::write_ulong(uint32_t data)
{
    fwrite(&data, sizeof(data), 1, file);
    data_size  += 4;
    update_data = true;
}

