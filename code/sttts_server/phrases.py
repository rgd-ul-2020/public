#!/usr/bin/python3

import sys

from abc import ABC, abstractmethod
from random import random

from words import *

def structure_1():
    nouns = Nouns()
    verbs = Intransitives()
    preps = Prepositions()
    adjcs = Adjectives()

    return 'The {} {} {} the {} {}.'.format(
        nouns.pop_random(),
        verbs.pop_random(),
        preps.pop_random(),
        adjcs.pop_random(),
        nouns.pop_random(),
    ).capitalize()


def structure_2():
    adjcs = Adjectives()
    nouns = Nouns()
    verbs = Transitives()

    return 'The {} {} {} the {}.'.format(
        adjcs.pop_random(),
        nouns.pop_random(),
        verbs.pop_random(),
        nouns.pop_random(),
    ).capitalize()


def structure_3():
    verbs = Transitives()
    nouns = Nouns()
    conjs = Conjunctions()

    return '{} the {} {} the {}.'.format(
        verbs.pop_random(),
        nouns.pop_random(),
        conjs.pop_random(),
        nouns.pop_random(),
    ).capitalize()

    
def structure_4():
    quest = QuestionAdverbials()
    auxis = Auxiliaries()
    nouns = Nouns()
    verbs = Transitives()
    adjcs = Adjectives()
    nouns = Nouns()

    return '{} {} the {} {} the {} {}.'.format(
        quest.pop_random(),
        auxis.pop_random(),
        nouns.pop_random(),
        verbs.pop_random(),
        adjcs.pop_random(),
        nouns.pop_random(),
    ).capitalize()


def structure_5():
    nouns = Nouns()
    verb1 = Transitives()
    nouns = Nouns()
    relts = RelativePronouns()
    verb2 = Intransitives()

    return 'The {} {} the {} {} {}.'.format(
        nouns.pop_random(),
        verb1.pop_random(),
        nouns.pop_random(),
        relts.pop_random(),
        verb2.pop_random(),
    ).capitalize()


if __name__ == '__main__':
    struct_type = [
        structure_1,
        structure_2,
        structure_3,
        structure_4,
        structure_5
    ]

    try:
        num_phrases = int(sys.argv[1])
    except:
        num_phrases = 100

#    print('\\documentclass[conference]{IEEEtran}')
#    print('\\begin{document}')
    for i in range(num_phrases):
        print(struct_type[int(len(struct_type) * random())]())
#        if i != 0 and i % 10 == 0:
#            print('\\par\\bigskip')
#        else:
#            print()
#    print('\\end{document}')
