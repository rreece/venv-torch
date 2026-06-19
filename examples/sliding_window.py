#!/usr/bin/env python3
"""
Longest substring without repeating characters

Given a string, find the length of the longest substring that contains no duplicate characters.

Input:  "abcabcbb"
Output: 3   # "abc"

Input:  "bbbbb"
Output: 1   # "b"

Input:  "pwwkew"
Output: 3   # "wke"
"""


def run_sliding_window(input_string):
    sub_length = 0
    i = 0
    cmap = dict()
    for j, char in enumerate(input_string):
        if char in cmap and cmap[char] >= i:
            i = cmap[char] + 1  # advance left pointer past previous occurrence
        cmap[char] = j          # always update last seen index
        sub_length = max(sub_length, j - i + 1)
    return sub_length


def main():
    input_string = "abcabcbb"
    result = run_sliding_window(input_string)
    print(result)


if __name__ == "__main__":
    main()
