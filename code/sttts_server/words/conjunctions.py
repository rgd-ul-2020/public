from words import Words
from random import random

class Conjunctions(Words):
    def pop_random(self):
        if (random() < 0.5):
            return 'and'
        return 'or'

