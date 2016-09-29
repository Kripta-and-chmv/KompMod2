import Tests
from Seq_Gen import *
import scipy
import math
import sympy
import scipy.stats
import random


def generate_coef(fileName):
    random.seed()
    with open(fileName, 'w') as f:
        for i in range(0, 3):
            f.write(str(random.randrange(1, 100)) + ' ')
        f.write(str(random.randrange(1, 100)))

def generate_period_more_then_100(input_name, length=1000):
    gen = SeqGenerator()
    while True:
        gen.read_file(input_name)
        gen.generate_sequence(length, 1)
        sequence = gen.get_sequence()

        period = Tests.find_period(sequence)
        with open('seq.txt', 'w') as f:
            for i in range(len(sequence)):
                f.write(str(sequence[i]) + ' ')

        if (period < 100):
            generate_coeff(input_name)
        else:     
            break
    return gen

def generate_successful_seq(file_name, intervals_amount, subseq): 
    while True:
        gen = generate_period_more_then_100(file_name)   

        length = len(seq)
        with open('parametres.txt', 'w') as f:
            is_test1_40 = Tests.test_1(seq[length - 40:], 0.05, f)
            is_test1_100 = Tests.test_1(seq[length - 100:], 0.05, f)

            is_test2_40 = Tests.test_2(seq[length - 40:], gen.get_mod(),
                0.05, intervals_amount, False, f)
            is_test2_100 = Tests.test_2(seq[length - 100:], gen.get_mod(),
                0.05, intervals_amount, False, f)
            is_test3_40 = Tests.test_3(seq[length - 40:], gen.get_mod(),
                0.05, subseq, intervals_amount, f)
            is_test3_100 = Tests.test_3(seq[length - 100:], gen.get_mod(),
                0.05, subseq, intervals_amount, f)
            is_anderson = Tests.anderson_darling_test(seq[length - period:], 
                gen.get_mod(), 0.05, f)
            is_chi2 = Tests.chisqr_test(seq[length - period:], gen.get_mod(),
                0.05, int(5 * math.log10(period)), False, f)
        
            passing_the_tests = is_test1_40 and is_test1_100 and \
                is_test2_40 and is_test2_100 and is_test3_40 and \
                is_test3_100 and is_chi2 and is_anderson

            if(passing_the_tests is True):
                break
            else:
                generate_coef(fileName)
    return gen


gen = generate_period_more_then_100('input.txt')

seq = gen.get_sequence()
period = Tests.find_period(seq)
intervals_amount = 10
subseq = 4
length = len(seq)

with open('parametres.txt', 'w') as f:
    test1_40 = Tests.test_1(seq[length - 40:], 0.05, f)
    test1_100 = Tests.test_1(seq[length - 100:], 0.05, f)
    test2_40 = Tests.test_2(seq[length - 40:], gen.get_mod(), 0.05,
        intervals_amount, True, f)
    is_test2_100 = Tests.test_2(seq[length - 100:], gen.get_mod(),
        0.05, intervals_amount, True, f)
    is_test3_40 = Tests.test_3(seq[length - 40:], gen.get_mod(), 0.05, subseq,
        intervals_amount, f)
    is_test3_100 = Tests.test_3(seq[length - 100:], gen.get_mod(), 0.05,
        subseq, intervals_amount, f)
    is_anderson = Tests.anderson_darling_test(seq[length - period:],
        gen.get_mod(), 0.05, f)
    is_chi2 = Tests.chisqr_test(seq[length - period:], gen.get_mod(), 0.05, 
        int(5 * math.log10(period)), True, f)
