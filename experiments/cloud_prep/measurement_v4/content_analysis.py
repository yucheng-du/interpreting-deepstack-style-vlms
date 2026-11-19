"""Factorial contrasts for native entry comparisons."""
import math
from itertools import product
from statistics import mean


def factorial_effects(values):
    expected = {''.join(x) for x in product('ab', repeat=3)}
    if set(values) != expected or not all(math.isfinite(x) for x in values.values()):
        raise ValueError('Need eight distinct finite content-direction scores')
    effects = {}
    def cell(indices):
        return ''.join('b' if i in indices else 'a' for i in range(3))
    for i in range(3):
        effects[f'single_{i}_at_others_a'] = values[cell({i})] - values['aaa']
        others = [j for j in range(3) if j != i]
        effects[f'average_switch_{i}'] = mean(values[cell({i}|{j for j,v in zip(others,bits) if v})]
            - values[cell({j for j,v in zip(others,bits) if v})] for bits in product((0,1),repeat=2))
    for i,j in ((0,1),(0,2),(1,2)):
        effects[f'pair_{i}{j}_at_other_a'] = values[cell({i,j})] - values[cell({i})] - values[cell({j})] + values['aaa']
    effects['all_b_minus_all_a'] = values['bbb'] - values['aaa']
    effects['triple_interaction'] = sum((-1)**(3-bits.count('b'))*value for bits,value in values.items())
    return effects
