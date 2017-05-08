# PyConcat
A Python library for performing concatenative synthesis tasks.

## Documentation

The reference documentation is here:-

http://pyconcat.readthedocs.io/en/latest/source/PyConcat.html

example.py shows an example of how to use it.

## Workflow

The basic pipeline for performing concatenative synthesis with this tool is as follows.

1. Segmentation
   * None, framewise FFTs, onsets, beats
2. two
   * MFCCs, spectral moments, loudness, f0, HPCPs
3. three
   * Brute force linear search, kDTree, Viterbi, k-best Viterbi decoding
