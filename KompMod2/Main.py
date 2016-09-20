from Tests import *
from SeqGen import *
import scipy
import scipy.stats

def RandomCoeff(fileName):
    with open(fileName, 'w') as f:
        for i in range(0, 3):
            f.write(str(random.randrange(1, 100)) + ' ')
        f.write(str(random.randrange(1, 100)))


gen = SeqGenerator()

while True:

    gen.Reading('input.txt')
    gen.GenerateSequence(2000, 1)
    sequence=gen.GetSequence()

    period = Tests.FindPeriod(sequence)
    print(period)
    if (period<100):
        RandomCoeff('input.txt')
    else:
        break 

valiable = Tests.Test_2(sequence, gen.mod, 0.05, 10)
print (valiable)

anderson = Tests.Anderson_Darling_test(sequence, 'norm')
print(anderson)