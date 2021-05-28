from random import random

class Words:
    def pop_random(self):
        selected = int(random() * len(self.words))
        return self.words.pop(selected)

