import math
import copy
import random
import matplotlib.pyplot as plt

def RandomCoeff(fileName):
    with open(fileName, 'w') as f:
        for i in range(0, 3):
            f.write(str(random.randrange(1, 100)) + ' ')
        f.write(str(random.randrange(1, 100)))

class Generator:
    x=0
    def __init__(self):
        self.a=0
        self.b=0
        self.c=0
        self.m=0
        self.sequence=[]

    def Reading(self, fileName):
        koef=[]
        with open(fileName, 'r') as f:
            fileStr=f.readline()
            koef=fileStr.split(' ')
        for i in range(len(koef)):
            koef[i]=int(koef[i])
        self.a, self.b, self.c, self.m = koef[0], koef[1], koef[2], koef[3]

    def GenerateSequence(self, length, x0):
        """Принимает на вход длину и начальный элемент.
        Записывает в self.sequence получившийся список-последовательность"""
        self.sequence=[]
        self.sequence.append(0)
        self.sequence.append(0)
        self.sequence.append(x0)

        for i in range(2, length+2):
            self.sequence.append( (self.a*self.sequence[i]+self.b*self.sequence[i-2]+self.c) % self.m)
        self.sequence=self.sequence[3:]

    def GetSequence(self):
        return self.sequence

class Tests:
    def FindPeriod(sequence):
        """На вход подаётся список-последовательность. 
        Возвращает число-период"""
        #период определяетя с конца последовательности
        #берётся последовательность из 3ех элементов и ищется первое её повторение
        length=len(sequence)
        a=list(reversed(sequence[length-3:length]))
        for i in range(length-4, -1, -1):
            if (a[0]==sequence[i]):
                if(i-2>-1):                    
                    if(a[1]==sequence[i-1] and a[2]==sequence[i-2]):
                        i+=1#добавляем единицу, т.к. i указывает на начало второго периода
                            #а должен на конец первого
                        break

        period = length-i     
        return period

    def Test1_Random(period):
        U=1.96
       
    def Test_2(seq, m):
        K=10
        #разбиваем на K интервалов
        intervals=[]
        intervals.append(0)
        interLength=m/K
        lastPoint=interLength
        for i in range(9):
            intervals.append(lastPoint)
            lastPoint+=interLength
        intervals.append(m)
        ###################

        n=40

        frequency=[0,0,0,0,0,0,0,0,0,0]
        for i in range(0, 40):
            if(seq[i]<intervals[1]):
                frequency[0]+=1/40
            elif(seq[i]<intervals[2]):
                frequency[1]+=1/40
            elif(seq[i]<intervals[3]):
                frequency[2]+=1/40
            elif(seq[i]<intervals[4]):
                frequency[3]+=1/40
            elif(seq[i]<intervals[5]):
                frequency[4]+=1/40
            elif(seq[i]<intervals[6]):
                frequency[5]+=1/40
            elif(seq[i]<intervals[7]):
                frequency[6]+=1/40
            elif(seq[i]<intervals[8]):
                frequency[7]+=1/40
            elif(seq[i]<intervals[9]):
                frequency[8]+=1/40
            elif(seq[i]<intervals[10]):
                frequency[9]+=1/40

        width=2

        plt.bar(intervals[:10], frequency, width)
        plt.xticks(intervals)
        plt.show()

        expValue40 = 0


        return intervals