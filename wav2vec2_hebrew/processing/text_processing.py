#!/usr/bin/env python3
# -*- coding: utf-8 -*-
SOFIT_MAP = {
    'ך': 'כ',
    'ם': 'מ',
    'ן': 'נ',
    'ף': 'פ',
    'ץ': 'צ',
}
SOFIT_MAP_INV = {v: k for k, v in SOFIT_MAP.items()}


def get_alephbet():
    return "".join(chr(i) for i in range(ord('א'), ord('ת') + 1))


def remove_sofits(text):
    return text.translate(str.maketrans(SOFIT_MAP))


def return_sofits(text):
    tokens = text.split()
    for idx, t in enumerate(tokens):
        if t[-1] in SOFIT_MAP_INV:
            tokens[idx] = t[:-1] + SOFIT_MAP_INV[t[-1]]
    return " ".join(tokens)


if __name__ == '__main__':
    print(get_alephbet())
    print(remove_sofits("ךםןףץ"))
    print(return_sofits("ממ ננ פפ צצ"))
