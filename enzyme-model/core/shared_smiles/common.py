"""
Common SMILES utilities — split, GELU, LayerNorm, etc.

Migrated from kcat/src/utils/common.py.
No internal imports — self-contained.
"""

import math
import torch
import torch.nn as nn


def split(sm):
    """
    Split SMILES into words. Care for Cl, Br, Si, Se, Na etc.
    input: A SMILES
    output: A string with space between words
    """
    arr = []
    i = 0
    while i < len(sm) - 1:
        if not sm[i] in ['%', 'C', 'B', 'S', 'N', 'R', 'X', 'L', 'A', 'M',
                         'T', 'Z', 's', 't', 'H', '+', '-', 'K', 'F']:
            arr.append(sm[i])
            i += 1
        elif sm[i] == '%':
            arr.append(sm[i:i + 3])
            i += 3
        elif sm[i] == 'C' and sm[i + 1] in 'laux':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'B' and sm[i + 1] in 'reai':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'S' and sm[i + 1] in 'ier':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'N' and sm[i + 1] in 'ai':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'R' and sm[i + 1] in 'ba':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'X' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'L' and sm[i + 1] == 'i':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'A' and sm[i + 1] in 'lsug':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'M' and sm[i + 1] in 'gn':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'T' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'Z' and sm[i + 1] == 'n':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 's' and sm[i + 1] in 'ie':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 't' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'H' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == '+' and sm[i + 1] in '234':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == '-' and sm[i + 1] in '234':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'K' and sm[i + 1] == 'r':
            arr.append(sm[i:i + 2])
            i += 2
        elif sm[i] == 'F' and sm[i + 1] == 'e':
            arr.append(sm[i:i + 2])
            i += 2
        else:
            arr.append(sm[i])
            i += 1
    if i == len(sm) - 1:
        arr.append(sm[i])
    return ' '.join(arr)
