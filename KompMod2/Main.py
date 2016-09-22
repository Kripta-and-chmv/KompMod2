from Tests import *
from SeqGen import *
import scipy
import scipy.stats

def RandomCoeff(fileName):
    random.seed()
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
       
leng = len(sequence)

Test1valiable = Tests.Test_1(sequence[:period], 0.05)
print(Test1valiable)

Test2valiable40 = Tests.Test_2(sequence[:40], gen.mod, 0.05, 10, True)
Test2valiable100 = Tests.Test_2(sequence[:100], gen.mod, 0.05, 10, True)
print (Test2valiable40 and Test2valiable100)

Test3valiable40 = Tests.Test_3(sequence[:40], gen.mod, 0.05, 4, 16)
Test3valiable100 = Tests.Test_3(sequence[:100], gen.mod, 0.05, 4, 16)
print (Test3valiable40 and Test3valiable100)



anderson = Tests.Anderson_Darling_test(sequence, gen.mod)
print(anderson)