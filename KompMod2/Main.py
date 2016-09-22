import Tests
from SeqGen import *
import scipy
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

    period = Tests.FindPeriod(sequence)
    with open('seq.txt', 'w') as f:
        for i in range(len(sequence)):
            f.write(str(sequence[i]) + ' ')
    
    print(period) 
    if (period<100):
        RandomCoeff('input.txt')
    else:
        Test1valiable40 = Tests.Test_1(sequence[len(sequence) - 40:], 0.05)
        Test1valiable100 = Tests.Test_1(sequence[len(sequence) - 100:], 0.05)

        Test2valiable40 = Tests.Test_2(sequence[len(sequence) - 40:], gen.mod, 0.05, 10, False)
        Test2valiable100 = Tests.Test_2(sequence[len(sequence) - 100:], gen.mod, 0.05, 10, False)

        Test3valiable40 = Tests.Test_3(sequence[len(sequence) - 40:], gen.mod, 0.05, 4, 16)
        Test3valiable100 = Tests.Test_3(sequence[len(sequence) - 100:], gen.mod, 0.05, 4, 16)

        anderson = Tests.Anderson_Darling_test(sequence[len(sequence) - period:], gen.mod)

        chi2 = Tests.ChiSqr_Test(sequence[len(sequence) - period:], gen.mod)
        if((Test1valiable40 and Test1valiable100 and Test2valiable40 and Test2valiable100 and Test3valiable40 and Test3valiable100 and chi2 and anderson) == True):
            break 
        else:
            RandomCoeff('input.txt')

Test1valiable40 = Tests.Test_1(sequence[len(sequence) - 40:], 0.05)
Test1valiable100 = Tests.Test_1(sequence[len(sequence) - 100:], 0.05)

Test2valiable40 = Tests.Test_2(sequence[len(sequence) - 40:], gen.mod, 0.05, 10, True)
Test2valiable100 = Tests.Test_2(sequence[len(sequence) - 100:], gen.mod, 0.05, 10, True)

Test3valiable40 = Tests.Test_3(sequence[len(sequence) - 40:], gen.mod, 0.05, 4, 16)
Test3valiable100 = Tests.Test_3(sequence[len(sequence) - 100:], gen.mod, 0.05, 4, 16)

anderson = Tests.Anderson_Darling_test(sequence[len(sequence) - period:], gen.mod)

chi2 = Tests.ChiSqr_Test(sequence[len(sequence) - period:], gen.mod)


print(Test1valiable40 and Test1valiable100)
print (Test2valiable40 and Test2valiable100)

print (Test3valiable40 and Test3valiable100)


print(anderson)
print(chi2)