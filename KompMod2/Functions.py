import math
import copy


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
        xnext=0
        xn=x0
        xnprev = 0
        xnprevprev = 0
        for i in range(length):
            xnext=(self.a*xn+self.b*xnprevprev+self.c) % self.m
            self.sequence.append(xnext)
            #self.sequence = self.sequence + str(xnext)
            xnprevprev, xnprev, xn = xnprev, xn, xnext
        #self.sequence=str(seq).strip('[]')
        
    def GetSequence(self):
        return self.sequence

class Tests:
    def FindPeriod(sequence):
        """На вход подаётся список-последовательность. 
        Возвращает число-период"""
        period=0
        
        a=sequence[0:3]
        for i in range(3, len(sequence)):
            if (a[0]==sequence[i]):
                if(a[1]==sequence[i+1] and a[2]==sequence[i+2]):
                    period=i
                    break

        return period

    def Test1_Random():
        min=0
        max=0