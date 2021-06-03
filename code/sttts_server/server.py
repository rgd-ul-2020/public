#!/usr/bin/python3 

import json
import random
import signal
import socket
import struct
import sys
import time
import datetime

from timeit import default_timer as timer

def log(text, *args, **kwargs):
    global first_log
    kwargs['file'] = sys.stdout
    print(('[{}] ' + text).format(datetime.datetime.now(), *args, **kwargs), file=sys.stdout)
    print(('[{}] ' + text).format(datetime.datetime.now(), *args, **kwargs), file=sys.stderr)

def get_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(('google.com', 1))
        ip = sock.getsockname()[0]
    except:
        ip = None
    finally:
        sock.close()
    return ip


def sigint_handler(sig, frame):
    global server
    log('Program aborted.')
    server.running = False
    server.sock.close()


def datagram(sock, buf, addr):
    sent = sock.sendto(buf, addr)
    log('Datagram to {}: {} bytes sent', addr, sent)
    time.sleep(0.5)


def pop_random(array):
    selected = int(random.random() * len(array))
    return array.pop(selected)


class Server:
    def __init__(self):
        self.queue = []
        self.payloads = [
            self.payload_command,
            self.payload_audio,
            self.payload_text,
        ]
        self.commands = [
            self.command_connect,
            self.command_handshake,
            self.command_disconnect,
            self.command_get_next,
            self.command_end_of_file,
            self.command_end_of_test,
        ]
        self.main_loop()

    def reset_queue(self):
        with open('sounds/index.json') as f:
            phrases = json.load(f)
        self.queue = []
        for out in ["pcm", "txt"]:#"spx", "ops", "dsp", "txt"]:
            for typ in ["s1", "s2", "s3", "s4", "s5"]:
                for i in range(5):
                    self.queue.append((out, pop_random(phrases[typ])))

    def main_loop(self):
        log('Server started. IP: {}', get_ip())

        signal.signal(signal.SIGINT, sigint_handler)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(5)
        self.sock.bind(('0.0.0.0', 8080))

        self.running   = True
        self.connected = None

        try:
            while self.running:
                try:
                    data, addr = self.sock.recvfrom(65536)
                    payload_type = struct.unpack('!i', data[0:4])[0]
                    self.payloads[payload_type](addr, data[4:])
                except socket.timeout:
                    pass
        finally:
            self.sock.close()

    def payload_command(self, addr, data):
        cmd = struct.unpack('!i', data)[0]
        self.commands[cmd](addr, data[4:])

    def payload_audio(self, data):
        log('Received audio: {}bytes', len(data))

    def payload_text(self, data):
        log('Received text: {}', data);

    def command_connect(self, addr, data):
        log('Connected to: {}', addr)
        datagram(self.sock, struct.pack('!ii', 0, 1), addr)
        self.connected = addr
        self.reset_queue()

    def command_handshake(self, addr, data):
        log('Connected')
        self.connected = addr
        self.reset_queue()
       
    def command_disconnect(self, addr, data):
        self.connected = None
        log('Disconnected')

    def command_get_next(self, addr, data):
        if len(self.queue) == 0:
            log('End of Test')
            log('')
            datagram(self.sock, struct.pack('!ii', 0, 5), addr)
            return

        selected = self.queue.pop(int(random.random() * len(self.queue)))
        log('Left: {}, Selected: {}', len(self.queue), selected, endl='')

        if selected[0] == 'pcm':
            total = 0
            with open('sounds/4_compressed/pcm_alaw/' + selected[1], 'rb') as f:
                f.seek(44)
                try:
                    while True:
                        rbytes = f.read(16000)
                        rsize  = len(rbytes)
                        total += rsize
                        if rsize <= 0:
                            break
                        datagram(self.sock, struct.pack('!i', 1) + rbytes, addr)
                except:
                    raise
            datagram(self.sock, struct.pack('!ii', 0, 4), addr)
        else:
            with open('sounds/4_compressed/deepspeech/' + selected[1] + '.txt') as f:
            #with open('sounds/A_phrases/' + selected[1] + '.txt') as f:
                line = f.readline()
            datagram(self.sock, struct.pack('!i'+str(len(line)+0)+'s', 2, line.encode()), addr)
        log('')

    def command_end_of_file(self, addr, data):
        log('CMD EOF')

    def command_end_of_test(self, addr, data):
        log('CMD EOT')

server = Server()
