#!/usr/bin/env python3
"""
Two Sum

Given an array of integers and a target, return the indices of the two numbers
that add up to the target. You may assume exactly one solution exists, and you
may not use the same element twice.

Input:  nums = [2, 7, 11, 15], target = 9
Output: (0, 1)   # because nums[0] + nums[1] = 2 + 7 = 9

Input:  nums = [3, 2, 4], target = 6
Output: (1, 2)

Aim for better than O(n²).
"""


def calc_two_sum(nums, target):
    n_nums = len(nums)
    nmap = dict()
    for i in range(n_nums):
        key = target - nums[i]
        if key in nmap:
            return (nmap[key], i)
        nmap[nums[i]] = i


def main():
    nums = [3, 2, 4]
    target = 6
    result = calc_two_sum(nums, target)
    print(result)


if __name__ == "__main__":
    main()
