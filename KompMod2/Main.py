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


gen = SeqGenerator()

while True:

    gen.read_file('input.txt')
    gen.generate_sequence(1000, 1)
    sequence = gen.get_sequence()
    # for i in range (len(sequence)):
    #    sequence[i] = random.randrange(0, gen.mod)

    period = Tests.find_period(sequence)
    with open('seq.txt', 'w') as f:
        for i in range(len(sequence)):
            f.write(str(sequence[i]) + ' ')

    if (period < 100):
        generate_coeff('input.txt')
    else:
        # Test1valiable40 = Tests.Test_1(sequence[len(sequence) - 40:], 0.05)
        # Test1valiable100 = Tests.Test_1(sequence[len(sequence) - 100:], 0.05)

        # Test2valiable40 = Tests.Test_2(sequence[len(sequence) - 40:],
        # gen.mod, 0.05, 10, False)
        # Test2valiable100 = Tests.Test_2(sequence[len(sequence) - 100:],
        # gen.mod, 0.05, 10, False)

        # Test3valiable40 = Tests.Test_3(sequence[len(sequence) - 40:],
        # gen.mod, 0.05, 4, 16)
        # Test3valiable100 = Tests.Test_3(sequence[len(sequence) - 100:],
        # gen.mod, 0.05, 4, 16)

        # anderson = Tests.Anderson_Darling_test(
        # sequence[len(sequence) - period:], gen.mod)

        # chi2 = Tests.ChiSqr_Test(sequence[len(sequence) - period:], gen.mod)
        # if((Test1valiable40 and Test1valiable100 and Test2valiable40 and
        # Test2valiable100 and Test3valiable40 and Test3valiable100 and chi2
        # and anderson) == True):
        break
        # else:
        #    RandomCoeff('input.txt')


intervals_amount = 10
subseq = 4
with open('parametres.txt', 'w') as f:
    Test1valiable40 = Tests.test_1(sequence[len(sequence) - 40:], 0.05, f)
    Test1valiable100 = Tests.test_1(sequence[len(sequence) - 100:], 0.05, f)
    Test2valiable40 = Tests.test_2(
        sequence[len(sequence) - 40:],
        gen.get_mod(),
        0.05,
        intervals_amount,
        False,
        f)
    Test2valiable100 = Tests.test_2(
        sequence[len(sequence) - 100:],
        gen.get_mod(),
        0.05,
        intervals_amount,
        False,
        f)
    Test3valiable40 = Tests.test_3(
        sequence[len(sequence) - 40:],
        gen.get_mod(),
        0.05,
        subseq,
        intervals_amount,
        f)
    Test3valiable100 = Tests.test_3(
        sequence[len(sequence) - 100:],
        gen.get_mod(),
        0.05,
        subseq,
        intervals_amount,
        f)
    anderson = Tests.anderson_darling_test(
        sequence[len(sequence) - period:], gen.get_mod(), f)
    chi2 = Tests.chisqr_test(
        sequence[len(sequence) - period:], gen.get_mod(), 0.05, 
        int(5 * math.log10(len(sequence))), f)
