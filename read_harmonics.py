import pandas as pd
from sympy import *
from mpmath import *
import numpy as np
import matplotlib.pyplot as plt
import re


def read_gfc():
    file1 = open('eigen-5s.gfc', 'r')
    count = 0

    while True:
        count += 1

        # Get next line from file
        line = file1.readline()

        fnd = line.find('end_of_head')
        if line.find('max_degree') >= 0:
            nums = line.split(' ')
            nums =[i for i in nums if i != '']
            max_degree = int(nums[1])
        # n  = int(nums[1])
        # if line is empty
        # end of file is reached
        if not line or fnd >= 0:
            break
        print("Line{}: {}".format(count, line.strip()))

    #file2 = open('geo.csv', 'w')
    #file2.write(key_string)
    koef = np.zeros((max_degree+1,max_degree+1))
    n = 0
    m = 0
    while m < max_degree and m < max_degree:
        count += 1

        # Get next line from file
        line = file1.readline()
        nums = line.split(' ')
        nums = [i for i in nums if i != '']
        n  = int(nums[1])
        m  = int(nums[2])
        cc = float(nums[3])
        ss = float(nums[4])
        if nums[0] != 'gfc' and nums[0] != 'gfct':
            continue
        koef[n][m] = cc
        koef[m-1][n] = ss

    file1.close()
    return koef
