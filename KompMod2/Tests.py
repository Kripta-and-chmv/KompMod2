import math
import copy
import random
import scipy
import scipy.stats
import matplotlib.pyplot as plt

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

    def Test1_Random(period, a):
        U=scipy.stats.norm.ppf(1-a/2)       

    def Test_2(seq, mod, alpha, intervalsAmount):                
        def CreateIntervals(mod, amount):
            intervals = []
            intervals.append(0)
            interLength = mod / amount
            lastPoint = interLength
            for i in range(amount - 1):
                intervals.append(lastPoint)
                lastPoint += interLength
            intervals.append(mod)
            return intervals
        #################
        def FindFrequency(amount, intervals, sequence):
            """находит частоту попаданий элементов из sequence в интервалы intervals"""
            frequency=[0,0,0,0,0,0,0,0,0,0]

            for i in range(0,amount):
                if(sequence[i] < intervals[1]):
                    frequency[0] += 1 / amount
                elif(sequence[i] < intervals[2]):
                    frequency[1] += 1 / amount
                elif(sequence[i] < intervals[3]):
                    frequency[2] += 1 / amount
                elif(sequence[i] < intervals[4]):
                    frequency[3] += 1 / amount
                elif(sequence[i] < intervals[5]):
                    frequency[4] += 1 / amount
                elif(sequence[i] < intervals[6]):
                    frequency[5] += 1 / amount
                elif(sequence[i] < intervals[7]):
                    frequency[6] += 1 / amount
                elif(sequence[i] < intervals[8]):
                    frequency[7] += 1 / amount
                elif(sequence[i] < intervals[9]):
                    frequency[8] += 1 / amount
                elif(sequence[i] < intervals[10]):
                    frequency[9] += 1 / amount
            return frequency   
        ##################
        def DrawHistogram(frequency, intervals, mod):
            width=mod
            plt.bar(intervals[:10], frequency, width)
            plt.xticks(intervals)
            plt.show()
        ##################
        def FindExpValue(sequence):
            expValue = 0.0
            for i in range(len(sequence)):
                expValue += sequence[i]
            expValue /= len(sequence)
            return expValue     
        ##################
        def FindVariance(sequence, expValue):
            variance = 0.0
            for i in range(len(sequence)):
                variance += (sequence[i] - expValue) ** 2
            variance /= len(sequence) - 1
            return variance
        #################
        def EvaluateFreq(freq, amount):
            """возвращает кол-во попавших в интервал частот"""
            def FindFreqIntervals(freq, amount):
                freqIntervals=[[],[]]
                for i in range(len(freq)):
                    k = (U / intervalsAmount) * math.sqrt((intervalsAmount - 1) / amount)
                    freqIntervals[0].append(freq[i] - k)
                    freqIntervals[1].append(freq[i] + k)
                return freqIntervals

            amountOfHits = 0
            intervals = FindFreqIntervals(freq, amount)
            P = 1 / intervalsAmount
            for i in range(len(freq)):
                if(intervals[0][i] <= P <= intervals[1][i]):
                    amountOfHits += 1
            return amountOfHits
        #################
        def EvaluateExpValue(expValue, variance, amount, mod):
            def FindExpValInterval(expValue, variance, amount):
                expValInterval = []
                k = U * math.sqrt(variance) / math.sqrt(amount)
                expValInterval.append(expValue-k)
                expValInterval.append(expValue+k)
                return expValInterval
            
            Mk = mod / 2
            expValueInterval = FindExpValInterval(expValue, variance, amount)
            hit = False
            if(expValueInterval[0] <= Mk <= expValueInterval[1]):
                hit=True
            
            return hit
        #################
        def EvaluateVariance(variance, amount, alpha, mod):
            def FindVarianceInterval(variance, amount, alpha):
                varianceInterval=[]
                varianceInterval.append((amount - 1) * variance / scipy.stats.chi2.ppf(1 - alpha / 2, amount - 1))
                varianceInterval.append((amount - 1) * variance / scipy.stats.chi2.ppf(alpha / 2, amount - 1))            
                return varianceInterval
            Dk = mod ** 2 /12
            varianceInterval=FindVarianceInterval(variance, amount, alpha)
            hit=False
            if(varianceInterval[0] <= Dk <= varianceInterval[1]):
                hit=True
            return hit
        #################

        U = scipy.stats.norm.ppf(1 - alpha / 2)
        #разбиваем на intervalsAmount интервалов

        intervals=CreateIntervals(mod, intervalsAmount)
        
        freq40 = FindFrequency(40, intervals, seq)
        DrawHistogram(freq40, intervals, mod / intervalsAmount)
        expValue40 = FindExpValue(seq[:40])
        variance40 = FindVariance(seq[:40], expValue40)
        
        freq100=FindFrequency(100, intervals, seq)
        DrawHistogram(freq100, intervals, mod / intervalsAmount)
        expValue100 = FindExpValue(seq[:100])
        variance100 = FindVariance(seq[:100], expValue100)

        freqGrade40 = EvaluateFreq(freq40, 40)
        expValueGrade40 = EvaluateExpValue(expValue40, variance40, 40, mod)
        variananceGrade40 = EvaluateVariance(variance40, 40, alpha, mod)

        freqGrade100 = EvaluateFreq(freq100, 100)
        expValueGrade100 = EvaluateExpValue(expValue100, variance100, 100, mod)
        variananceGrade100 = EvaluateVariance(variance100, 100, alpha, mod)

        return expValueGrade40 and variananceGrade40 and expValueGrade100 and variananceGrade100