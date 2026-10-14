#!/usr/bin/env python3
"""Retired mixed-scale top3 sampler. Historical artifacts remain diagnostic only."""


def main():
    raise RuntimeError(
        'Legacy max(alpha,label_map,bbox) sampling/scoring is retired. '
        'Use tools/build_fi1_margin_preparation.py for frozen GT-free source/scene '
        'sampling, multi-class hypotheses and explicit partition-bound score actions. '
        'Old local labels must not be reused for training.'
    )


if __name__ == '__main__':
    main()
