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

Test1valiable40, outTest1_40 = Tests.Test_1(sequence[len(sequence) - 40:], 0.05)
Test1valiable100, outTest1_100 = Tests.Test_1(sequence[len(sequence) - 100:], 0.05)

outTest2_40 = []
outTest2_100 =[]

intervalsAmount = 10
subseq = 4

Test2valiable40, outTest2_40 = Tests.Test_2(sequence[len(sequence) - 40:], gen.mod, 0.05, intervalsAmount, True)
Test2valiable100, outTest2_100 = Tests.Test_2(sequence[len(sequence) - 100:], gen.mod, 0.05, intervalsAmount, True)

outTest3_40 = []
outTest3_100 =[]

Test3valiable40, outTest3_40 = Tests.Test_3(sequence[len(sequence) - 40:], gen.mod, 0.05, subseq, intervalsAmount)
Test3valiable100, outTest3_100 = Tests.Test_3(sequence[len(sequence) - 100:], gen.mod, 0.05, subseq, intervalsAmount)

anderson, outAnderson = Tests.Anderson_Darling_test(sequence[len(sequence) - period:], gen.mod)

chi2, outChi2 = Tests.ChiSqr_Test(sequence[len(sequence) - period:], gen.mod, 0.05, int(5 * math.log10(len(sequence))))

s = list(reversed(sequence))
##########################ГОВНОКОД################################################
with open('period.txt', 'w') as f:
    for i in range(period):
        f.write(str(s[i]) + ' ')

with open('parametres.txt', 'w') as f:
    if (period==len(sequence)):
        f.write('Длина периода: равна длине последовательноси\n')
    f.write('Длина периода: ' + str(period) + '\n')
    f.write('Тест №1: ' + str(Test1valiable40 and Test1valiable100) + '\n')
    f.write('\t 40 ' + 'число перестановок: ' + str(outTest1_40[0]) + '\n')
    f.write('\t 40 ' + 'мат ожидание: ' + str(outTest1_40[1]) + '\n')
    f.write('\t 40 ' + 'интервал: ' + str(outTest1_40[2][0]) + ' - ' + str(outTest1_100[2][1]) + '\n')
    f.write('\t 100 ' + 'число перестановок: ' + str(outTest1_100[0]) + '\n')
    f.write('\t 100 ' + 'мат ожидание: ' + str(outTest1_100[1]) + '\n')
    f.write('\t 100 ' + 'интервал: ' + str(outTest1_100[2][0]) + ' - ' + str(outTest1_100[2][1]) + '\n\n')

    f.write('Тест №2: ' + str(Test2valiable40 and Test2valiable100) + '\n')
    f.write('\t' + '40: ' + 'мат ожидание: ' + str(outTest2_40[1][1]) + '. Интервал: ' + str(outTest2_40[1][0][0]) + ' - ' + str(outTest2_40[1][0][1]) + '\n')
    f.write('\t' + '40: ' + 'дисперсия: ' + str(outTest2_40[2][1]) + '. Интервал: ' + str(outTest2_40[2][0][0]) + ' - ' + str(outTest2_40[2][0][1]) + '\n')
    f.write('\t' + '40:' + ' вероятность: ' + str(outTest2_40[0][1]) + '\n\t интервалы: \n')    
    for i in range(intervalsAmount): 
        f.write('\t\t' + str(outTest2_40[0][0][0][i]) + ' - ' + str(outTest2_40[0][0][1][i]) + '\n')    
    f.write('\t' + '100: ' + 'мат ожидание: ' + str(outTest2_100[1][1]) + '. Интервал: ' + str(outTest2_100[1][0][0]) + ' - ' + str(outTest2_100[1][0][1]) + '\n')
    f.write('\t' + '100: ' + 'дисперсия: ' + str(outTest2_100[2][1]) + '. Интервал: ' + str(outTest2_100[2][0][0]) + ' - ' + str(outTest2_100[2][0][1]) + '\n')
    f.write('\t' + '100:' + ' вероятность: ' + str(outTest2_100[0][1]) + '\n\t интервалы: \n')    
    for i in range(intervalsAmount): 
        f.write('\t\t' + str(outTest2_100[0][0][0][i]) + ' - ' + str(outTest2_100[0][0][1][i]) + '\n')    
    
    f.write('\nТест Андерсона: ' + str(anderson) + '\n')
    f.write('\t статистика: ' + str(outAnderson[0]) + '\n')
    f.write('\t критическое значение: ' + str(outAnderson[1]) + '\n\n')

    f.write('Тест Хи квадрат: ' + str(chi2) + '\n')
    f.write('\t статистика: ' + str(outChi2[0]) + '\n')
    f.write('\t критическое значение: ' + str(outChi2[1]) + '\n\n')

with open('test3.txt', 'w') as f:
    f.write('======================== 40 элементов =====================================')
    for i in range(subseq):
        f.write('\n\n__________________ПОДПОСЛЕДОВАТЕЛЬНОСТЬ № ' + str(i + 1) + '______________________________\n')
        f.write('Тест №1 для 40 элементов: \n')
        f.write('\tчисло перестановок: ' + str(outTest3_40[0][i][0]) + '\n')
        f.write('\tмат ожидание: ' + str(outTest3_40[0][i][1])+ '\n')
        f.write('\tинтервал: ' + str(outTest3_40[0][i][2][0]) + ' - ' + str(outTest3_40[0][i][2][1]) + '\n\n')

        f.write('Тест №2 для 40 элементов:\n')
        f.write('\tмат ожидание: ' + str(outTest3_40[1][i][1][1]) + '. Интервал: ' + str(outTest3_40[1][i][1][0][0]) + ' - ' + str(outTest3_40[1][i][1][0][1]) + '\n')
        f.write('\tдисперсия: ' + str(outTest3_40[1][i][2][1]) + '. Интервал: ' + str(outTest3_40[1][i][2][0][0]) + ' - ' + str(outTest3_40[1][i][2][0][1]) + '\n')
        f.write('\tвероятность: ' + str(outTest3_40[1][i][0][1]) + '\n\t интервалы: \n')    
        for j in range(intervalsAmount): 
            f.write('\t\t' + str(outTest3_40[1][i][0][0][0][j]) + ' - ' + str(outTest3_40[1][i][0][0][1][j]) + '\n')    
        
    f.write('\n\n======================== 100 элементов =====================================')
        
    for i in range(subseq):
        f.write('\n\n__________________ПОДПОСЛЕДОВАТЕЛЬНОСТЬ № ' + str(i + 1) + '______________________________\n')
        f.write('Тест №1: \n')
        f.write('\tчисло перестановок: ' + str(outTest3_100[0][i][0]) + '\n')
        f.write('\tмат ожидание: ' + str(outTest3_100[0][i][1])+ '\n')
        f.write('\tинтервал: ' + str(outTest3_100[0][i][2][0]) + ' - ' + str(outTest3_100[0][i][2][1]) + '\n\n')

        f.write('Тест №2:\n')
        f.write('\tмат ожидание: ' + str(outTest3_100[1][i][1][1]) + '. Интервал: ' + str(outTest3_100[1][i][1][0][0]) + ' - ' + str(outTest3_100[1][i][1][0][1]) + '\n')
        f.write('\tдисперсия: ' + str(outTest3_100[1][i][2][1]) + '. Интервал: ' + str(outTest3_100[1][i][2][0][0]) + ' - ' + str(outTest3_100[1][i][2][0][1]) + '\n')
        f.write('\tвероятность: ' + str(outTest3_100[1][i][0][1]) + '\n\t интервалы: \n')    
        for j in range(intervalsAmount): 
            f.write('\t\t' + str(outTest3_100[1][i][0][0][0][j]) + ' - ' + str(outTest3_100[1][i][0][0][1][j]) + '\n')    
