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
        """Принимает на вход длину и начальное приближение.
        Возвращает список-последовательность"""
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
    #def FindPeriod(sequence):
    #    """На вход подаётся строковое представленеие последовательности. 
    #    Возвращает строку-период"""
    #    period=""
    #    for i in range(2, len(sequence)):
    #        testSequence=sequence[:i]
    #        index=sequence.find(testSequence, i)#находим первое вхождение тестовой 
    #                                            #последовательности в оставшейся строке
    #        if(i==index):#если первое вхождение следует после тестовой последовательности, то период найден
    #            break

    #        if(i>len(sequence)/2):#если предполагаемый период больше половины последовательности, то проверяем,
    #                              #не является ли оставшаяся часть последовательности частью периода
    #            index=testSequence.find(sequence[i:], 0)
    #            if(index==0):
    #                break

    #    #если значение периода не изменилось, то за период принимаем длину последовательности 
    #    period=testSequence
    #    return period

    def FindPeriodFloyd(gen, a0):
        def TakeElem(k):
            gen.GenerateSequence(1, k)
            seq=gen.GetSequence()
            return seq[0]
        
        a=a0
        b=TakeElem(a)

        while a!=b:
            a=TakeElem(a)
            c=TakeElem(b)
            b=TakeElem(c)
        b=TakeElem(a)
        t=1
        while a!=b:
            b=TakeElem(b)
            t+=1
        return t
        

    def Test1_Random():
        min=0
        max=0