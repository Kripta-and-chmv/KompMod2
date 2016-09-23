import Tests
from SeqGen import *
import scipy
import math
import sympy
import scipy.stats
import random


def RandomCoeff(fileName):
    random.seed()
    with open(fileName, 'w') as f:
        for i in range(0, 3):
            f.write(str(random.randrange(1, 100)) + ' ')
        f.write(str(random.randrange(1, 100)))


gen = SeqGenerator()

while True:

    gen.Reading('input.txt')
    gen.GenerateSequence(1000, 1)
    sequence=gen.GetSequence()
    #for i in range (len(sequence)):
    #    sequence[i] = random.randrange(0, gen.mod)

    period = Tests.FindPeriod(sequence)
    with open('seq.txt', 'w') as f:
        for i in range(len(sequence)):
            f.write(str(sequence[i]) + ' ')
   
    if (period<100):
        RandomCoeff('input.txt')
    else:
        #Test1valiable40 = Tests.Test_1(sequence[len(sequence) - 40:], 0.05)
        #Test1valiable100 = Tests.Test_1(sequence[len(sequence) - 100:], 0.05)

        #Test2valiable40 = Tests.Test_2(sequence[len(sequence) - 40:], gen.mod, 0.05, 10, False)
        #Test2valiable100 = Tests.Test_2(sequence[len(sequence) - 100:], gen.mod, 0.05, 10, False)

        #Test3valiable40 = Tests.Test_3(sequence[len(sequence) - 40:], gen.mod, 0.05, 4, 16)
        #Test3valiable100 = Tests.Test_3(sequence[len(sequence) - 100:], gen.mod, 0.05, 4, 16)

        #anderson = Tests.Anderson_Darling_test(sequence[len(sequence) - period:], gen.mod)

        #chi2 = Tests.ChiSqr_Test(sequence[len(sequence) - period:], gen.mod)
        #if((Test1valiable40 and Test1valiable100 and Test2valiable40 and Test2valiable100 and Test3valiable40 and Test3valiable100 and chi2 and anderson) == True):
            break 
        #else:
        #    RandomCoeff('input.txt')


intervalsAmount = 10
subseq = 4
with open('parametres.txt', 'w') as f:
    Test1valiable40 = Tests.Test_1(sequence[len(sequence) - 40:], 0.05, f)
    Test1valiable100 = Tests.Test_1(sequence[len(sequence) - 100:], 0.05, f)
    Test2valiable40 = Tests.Test_2(sequence[len(sequence) - 40:], gen.mod, 0.05, intervalsAmount, False, f)
    Test2valiable100 = Tests.Test_2(sequence[len(sequence) - 100:], gen.mod, 0.05, intervalsAmount, False, f)
    Test3valiable40 = Tests.Test_3(sequence[len(sequence) - 40:], gen.mod, 0.05, subseq, intervalsAmount, f)
    Test3valiable100 = Tests.Test_3(sequence[len(sequence) - 100:], gen.mod, 0.05, subseq, intervalsAmount, f)
    anderson = Tests.Anderson_Darling_test(sequence[len(sequence) - period:], gen.mod, f)
    chi2 = Tests.ChiSqr_Test(sequence[len(sequence) - period:], gen.mod, 0.05, int(5 * math.log10(len(sequence))), f)