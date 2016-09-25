import math
import numpy
import copy
import random
import sympy
import scipy
import scipy.stats
import matplotlib.pyplot as plt


def find_period(sequence):
    """На вход подаётся список-последовательность.

    Возвращает число-период

    """
    # период определяетя с конца последовательности
    # берётся последовательность из 3ех элементов и ищется первое её повторение
    length = len(sequence)
    a = list(reversed(sequence[length - 3: length]))
    for i in range(length - 4, -1, -1):
        if (a[0] == sequence[i]):
            if(i - 2 > -1):
                if(a[1] == sequence[i - 1] and a[2] == sequence[i - 2]):
                    # добавляем единицу, т.к. i указывает на начало второго периода
                    i += 1
                    # а должен на конец первого
                    break

    period = length - i
    return period


def test_1(sequence, alpha, wfile):
    """Подаётся последовательность seq и уровень значимости alpha. Помимо
    этого, происходит запись результатов теста в файл wfile.

    Возвращает значение bool - результат прохождения теста.

    """

    def WritingInFile(wfile, length, hit, interval):
        """В файл wfile записывается размер выборки length, успешность
        прохождения теста hit, мат ожидание length/2 и доверительный интервал
        interval."""
        wfile.write(
            '============================== Тест 1 ==============================\n\n')
        wfile.write('Размер выборки: %s\n' % (length))
        wfile.write('Успешность прохождения: %s\n' % (hit))
        wfile.write('Математическо ожидание: %s\n' % (length / 2))
        wfile.write(
            'Доверительный интервал: [%s, %s]\n\n' %
            (interval[0], interval[1]))
    # Q - оценка количества перестановок
    Q = 0
    for i in range(len(sequence) - 1):
        if(sequence[i] > sequence[i + 1]):
            Q += 1
    # строим ассимпотический доверителный интервал для Q
    # interval[0] - нижняя граница, interval[1] - верхняя
    U = scipy.stats.norm.ppf(1 - alpha / 2)
    delta = U * math.sqrt(len(sequence)) / 2
    interval = []
    interval.append(Q - delta)
    interval.append(Q + delta)

    # проверяем попадает ли мат ожидание числа перестановок,
    # P/2 или половина длины последовательности, в доверительный интервал
    hit = False
    if(interval[0] <= len(sequence) / 2 <= interval[1]):
        hit = True

    WritingInFile(wfile, len(sequence), hit, interval)

    return hit


def test_2(seq, mod, alpha, intervalsAmount, drawingGraph, wfile):
    """Подаётся последоватеьность seq, mod - размерность алфавита,
        alpha - уровень значимости, intervalAmount - количество создаваемых интервалов,
        drawingGraph - bool значение, указывающее, нужно ли рисовать гистограмму,
        wfile - файл, куда будет записываться результат выполнения теста.
        возвращается значение bool - результат попадания попадания теоретических мат. ожиданий и дисперсий
        в построенные доверительные интервалы"""
    def CreateIntervals(mod, intervalsAmount):
        """разбиваем отрезок от 0 до mod на amount интервалов.

        возвращается массив со значениями всех границ интервалов

        """
        intervals = []
        intervals.append(0)
        interLength = mod / intervalsAmount
        lastPoint = interLength
        for i in range(intervalsAmount - 1):
            intervals.append(lastPoint)
            lastPoint += interLength
        intervals.append(mod)
        return intervals

    def FindFrequency(intervals, sequence):
        """находит частоту попаданий элементов из sequence в интервалы
        intervals."""
        frequency = numpy.zeros(len(intervals) - 1)
        length = len(sequence)
        piece = 1 / length
        for i in range(length):
            for j in range(len(intervals) - 1):
                if (intervals[j] <= sequence[i] < intervals[j + 1]):
                    frequency[j] += piece

        return frequency

    def DrawHistogram(frequency, intervals):
        """Рисует гистограмму частот frequency на интервалах intervals."""

        # ширина стобца - размер алфавита делаится на количество интервалов
        width = intervals[len(intervals) - 1] / (len(intervals) - 1)

        plt.bar(intervals[:len(intervals) - 1], frequency, width)
        plt.title('Frecuency Histogram')
        plt.xlabel('intervals')
        plt.ylabel('relative frequency')
        plt.xticks(intervals)
        plt.show()

    def FindExpValue(sequence):
        """Возвращает мат ожидание выборки sequence."""
        expValue = 0.0
        for i in range(len(sequence)):
            expValue += sequence[i]
        expValue /= len(sequence)
        return expValue

    def FindVariance(sequence, expValue):
        """возвращает дисперсию выборки sequence с мат ожиданием expValue."""
        variance = 0.0
        for i in range(len(sequence)):
            variance += (sequence[i] - expValue) ** 2
        variance /= len(sequence) - 1
        return variance

    def EvaluateFreq(freq, seqLength, U, intervalsAmount):
        """принимает список частот freq, длину последвательности seqLength,
        квантиль U, кол-во интервалов intervalsAmount.

        Возвращает частоту интервалы для каждой частоты, кол-во попавших
        в построенные ассимптотические доверительные интервалы теоретич.
        частот

        """

        def FindFreqIntervals(freq, seqLength, intervalsAmount):
            """возвращает список ассимптотических доверительных интервалов для
            каждого значения из списка частот freq.

            каждый i-ый элемент freqIntervals[0] - няжняя граница дов интервала для freq[i],
            каждый i-ый элемент freqIntervals[1] - верхняя граница дов интервала для freq[i]

            """
            freqIntervals = [[], []]
            for i in range(len(freq)):
                delta = (U / intervalsAmount) * \
                    math.sqrt((intervalsAmount - 1) / seqLength)
                freqIntervals[0].append(freq[i] - delta)
                freqIntervals[1].append(freq[i] + delta)
            return freqIntervals

        # количество частот, попавших в доверительные интервалы
        amountOfHits = 0

        intervals = FindFreqIntervals(freq, seqLength, intervalsAmount)
        P = 1 / intervalsAmount
        for i in range(len(freq)):
            if(intervals[0][i] <= P <= intervals[1][i]):
                amountOfHits += 1

        return P, intervals, amountOfHits

    def EvaluateExpValue(expValue, variance, seqLength, mod, U):
        """Принимает мат ожидание expValue, дисперсию variance, длину выборки
        seqLength, размер алфавита mod, квантиль U. Возвращает теоретическое
        мат ожидание, интервал для построенного мат ожидания,

        bool значение - результат попадания теоретического мат ожидания в построенныей ассимптотический доверительный интервал

        """
        def FindExpValInterval(expValue, variance, seqLength, U):
            """Возвращает ассимптотический доверительный интервал для мат
            ожидания expValue.

            expValInterval[0] - нижняя граница. expValInterval[1] - верхняя граница

            """
            expValInterval = []
            delta = U * math.sqrt(variance) / math.sqrt(seqLength)
            expValInterval.append(expValue - delta)
            expValInterval.append(expValue + delta)
            return expValInterval

        Mk = mod / 2
        expValueInterval = FindExpValInterval(expValue, variance, seqLength, U)
        hit = False
        if(expValueInterval[0] <= Mk <= expValueInterval[1]):
            hit = True

        return Mk, expValueInterval, hit

    def EvaluateVariance(variance, seqLength, alpha, mod):
        """принимает дисперсию variance, длину выборки sqeLength, уровень
        значимости alpha, размер алфавита mod. Возвращает теоретическую
        дисперсию, интервал для построенной дисперсии,

        bool значение - результат попадания теоретической дисперсии в построенныей ассимптотический доверительный интервал

        """
        def FindVarianceInterval(variance, seqLength, alpha):
            """Возвращает ассимптотический доверительный интервал для дисперсии
            variance.

            variancInterval[0] - нижняя граница. variancInterval[1] - верхняя граница.

            """
            varianceInterval = []
            varianceInterval.append(
                (seqLength -
                 1) *
                variance /
                scipy.stats.chi2.ppf(
                    1 -
                    alpha /
                    2,
                    seqLength -
                    1))
            varianceInterval.append(
                (seqLength -
                 1) *
                variance /
                scipy.stats.chi2.ppf(
                    alpha /
                    2,
                    seqLength -
                    1))
            return varianceInterval

        Dk = mod ** 2 / 12
        varianceInterval = FindVarianceInterval(variance, seqLength, alpha)
        hit = False
        if(varianceInterval[0] <= Dk <= varianceInterval[1]):
            hit = True

        return Dk, intervals, hit

    def WritingInFile(wfile, length, intervalsAmount, hitsAmount, teorFreq,
                      freqInterv, expVhit, teorexpV, expVinterval, varHit, teorVar, varInterval):
        """В файл wfile записывается размер выборки length, количество
        построенных интервалов, теоретическая вероятность попадания элемента в
        интервал, количество попавших в интервалы элементов, интервалы для
        каждой частоты попадания в интервал, результат попадания мат ожидания в
        интервал, теоретичесоке мат ожидание, доверительный интервал для мат
        ожидания, результат попадания дисперсии в инервал, теоретическая
        дисперсия, доверительный интервал для дисперсии."""
        wfile.write(
            '============================== Тест 2 ==============================\n\n')
        wfile.write('Размер выборки: %s\n' % (length))
        wfile.write('Количество интервалов: %s\n' % (intervalsAmount))
        wfile.write(
            'Количество попаданий теор частоты в интервалы: %s\n' %
            (hitsAmount))
        wfile.write('Теоретическая частота попаданий: %s\n' % (teorFreq))
        wfile.write(
            'Доверительный интервалы для каждой вычесленной частоты: \n')
        for i in range(intervalsAmount):
            wfile.write('\t[%s, %s]\n' % (freqInterv[0][i], freqInterv[1][i]))

        wfile.write(
            'Попадание теоретич. математическое ожидание в доверительный интервал: %s\n' %
            (expVhit))
        wfile.write('Теоретическое математическое ожидание: %s\n' % (teorexpV))
        wfile.write(
            'Доверительный интервал для мат ожидания: [%s, %s]\n\n' %
            (expVinterval[0], expVinterval[1]))

        wfile.write(
            'Попадание теоретич. дисперсии в доверительный интервал: %s\n' %
            (varHit))
        wfile.write('Теоретическая дисперсия: %s\n' % (teorVar))
        wfile.write(
            'Доверительный интервал для дисперсии: [%s, %s]\n\n' %
            (varInterval[0], varInterval[1]))

    # вычисляем квантиль уровня 1 - alpha/2 нормального распределения с мат.
    # ожид. 0 и среднеквадратич. отклонением 1
    U = scipy.stats.norm.ppf(1 - alpha / 2)
    # разбиваем на intervalsAmount интервалов

    intervals = CreateIntervals(mod, intervalsAmount)

    length = len(seq)

    freq = FindFrequency(intervals, seq)
    if (drawingGraph is True):
        DrawHistogram(freq, intervals)
    expValue = FindExpValue(seq)
    variance = FindVariance(seq, expValue)

    teorFreq, freqInterv, hitsAmount = EvaluateFreq(
        freq, length, U, intervalsAmount)
    teorexpV, expVinterval, expVhit = EvaluateExpValue(
        expValue, variance, length, mod, U)
    teorVar, varInterval, varHit = EvaluateVariance(
        variance, length, alpha, mod)

    WritingInFile(
        wfile,
        length,
        intervalsAmount,
        hitsAmount,
        teorFreq,
        freqInterv,
        expVhit,
        teorexpV,
        expVinterval,
        varHit,
        teorVar,
        varInterval)

    return expVhit and varHit,
    intervalsAmount,
    hitsAmount,
    teorFreq,
    freqInterv,
    expVhit,
    teorexpV,
    expVinterval,
    varHit,
    teorVar,
    varInterval
    


def test_3(seq, mod, alpha, subseqsAmount, intervalsAmount, wfile):
    """Принимает выборку seq, размер алфавита mod, уровень значимости alpha,
    количество подпоследовательностей subseqAmount, количество интервалов для
    второго теста intervalsAmount Возвращает False, если хотя бы одна
    подпоследовательность не прошла Test 1 и Test 2, иначе True."""

    def MakingSubseqs(seq, subseqsAmount):
        """принимает выборку seq и кол-во подпоследовательностей subseqsAmount.

        возвращает список подпоследовательностей, содержащий все
        элементы из seq

        """
        seqLength = len(seq)
        t = int((seqLength - 1) / subseqsAmount)
        seqArray = []

        for i in range(subseqsAmount):
            s = []
            for j in range(t + 1):
                s.append(seq[j * subseqsAmount + i])
            seqArray.append(s)
        return seqArray

    seqArray = MakingSubseqs(seq, subseqsAmount)
    hit_Test1 = True
    hit_Test2 = True

    wfile.write(
        '============================== Тест 3 ==============================\n\n')
    wfile.write('Размер выборки: %s\n' % (seq))
    wfile.write('Количество подпоследовательностей: %s\n\n' % (subseqsAmount))
    for i in range(len(seqArray)):
        wfile.write(
            '\t\tПрохождение тестов 1 и 2 для  %s подпоследовательности\n\n' %
            (i + 1))
        hit = test_1(seqArray[i], alpha, wfile)
        if (hit is False):
            hit_Test1 = False

        hit = test_2(seqArray[i], mod, alpha, intervalsAmount, False, wfile)
        if (hit is False):
            hit_Test2 = False

    return hit_Test1 and hit_Test2


def anderson_darling_test(_sequence, mod, wfile):
    """Тест андерсона на равномерное распределение. Принимает
    последовательность и размер алфавита.

    Возвращает bool значение - результат прохождения теста

    """
    def UniformDistrFunc(x, mod):
        """функция равномерного распредленения.

        принимает x - элемент выборки, mod - верхняя граница

        """
        a = 0
        b = mod
        if (x < a):
            return 0
        elif (a <= x < b):
            return (x - a) / (b - a)
        elif (x >= b):
            return 1

    def WritingInFile(wfile, length, S, Scrit, hit):
        """В файл wfile записывается размер выборки length, значение статистики
        и критическое значение."""
        wfile.write(
            '============================== Тест Андерсона ==============================\n\n')
        wfile.write('Размер выборки: %s\n' % (length))
        wfile.write('Успешность прохождения: %s\n' % (hit))
        wfile.write('Значение статистики: %s\n' % (S))
        wfile.write('Критическое значение: %s\n\n' % (Scrit))

    sequence = []
    sequence = sorted(copy.copy(_sequence))

    addition = 0
    length = len(sequence)

    for i in range(1, length + 1):
        F = UniformDistrFunc(sequence[i - 1], mod)
        if(F == 0):
            return False, [999999999999, 2.4924]
        addition += (2 * i - 1) * math.log(F) / (2 * length)
        addition += (1 - (2 * i - 1) / (2 * length)) * math.log(1 - F)

    S = -len(sequence) - 2 * addition
    criticalValue = 2.4924

    hit = False

    if (S <= criticalValue):
        hit = True

    WritingInFile(wfile, len(sequence), S, criticalValue, hit)

    return hit


def chisqr_test(sequence, mod, alpha, intervalsAmount, wfile):
    """Нв вход подаются выюорка sequence, размерность алфавита mod.

    Возвращается значение bool - результат прохождения теста

    """
    def CreateIntervals(mod, intervalsAmount):
        """разбиваем отрезок от 0 до mod на amount интервалов.

        возвращается массив со значениями всех границ интервалов

        """
        intervals = []
        intervals.append(0)
        interLength = mod / intervalsAmount
        lastPoint = interLength
        for i in range(intervalsAmount - 1):
            intervals.append(lastPoint)
            lastPoint += interLength
        intervals.append(mod)
        return intervals

    def FindHitsAmount(intervals, sequence):
        """находит количество элементов sequence, попадающих в каждые интервалы
        intervals."""
        frequency = numpy.zeros(len(intervals) - 1)
        length = len(sequence)
        for i in range(length):
            for j in range(len(intervals) - 1):
                if (intervals[j] <= sequence[i] < intervals[j + 1]):
                    frequency[j] += 1

        return frequency

    def FindIntervalsPorbability(intervals, a, b):
        """находим вероятность попадания сл величины в интервалы intervals при
        равномерном распределении. возвращаем список вероятностей.

        a - нижняя граница равномерного распределения, b - верхняя

        """
        probabil = []
        for i in range(len(intervals) - 1):
            probabil.append((intervals[i + 1] - intervals[i]) / (b - a))
        return probabil

    def DrawHistogram(frequency, intervals):
        """Рисует гистограмму частот frequency на интервалах intervals."""

        # ширина стобца - размер алфавита делаится на количество интервалов
        width = intervals[len(intervals) - 1] / (len(intervals) - 1)

        plt.bar(intervals[:len(intervals) - 1], frequency, width)
        plt.title('Chi2 Histogram')
        plt.xlabel('intervals')
        plt.ylabel('hits amount')
        plt.xticks(intervals)
        plt.show()

    def WritingInFile(wfile, length, S, Scrit, hit, intervAmount):
        """В файл wfile записывается размер выборки length, значение статистики
        и критическое значение."""
        wfile.write(
            '============================== Тест Андерсона ==============================\n\n')
        wfile.write('Количество интервалов: %s\n\n' % (intervAmount))
        wfile.write('Успешность прохождения: %s\n' % (hit))
        wfile.write('Значение статистики: %s\n' % (S))
        wfile.write('Критическое значение: %s\n\n' % (Scrit))

    intervals = CreateIntervals(mod, intervalsAmount)
    hitsAmount = FindHitsAmount(intervals, sequence)

    probabil = FindIntervalsPorbability(intervals, 0, mod)

    DrawHistogram(hitsAmount, intervals)

    # вычисляется статистика
    addition = 0
    for i in range(intervalsAmount):
        addition += (hitsAmount[i] / len(sequence) -
                     probabil[i]) ** 2 / probabil[i]
    S = len(sequence) * addition

    x = sympy.Symbol('x')

    r = 5

    #Schi2 = sympy.integrate(x ** (r / 2 - 1) * sympy.exp(-x / 2),  (x, S, sympy.oo)) / (2 ** (r / 2) * math.gamma(int(r / 2)))

    #hitChi2 = False
    # if (Schi2 > alpha):
    #    hitChi2 = True

    S_crit = 18.307
    hit = False
    if(S <= S_crit):
        hit = True

    WritingInFile(wfile, len(sequence), S, S_crit, hit, intervalsAmount)

    return hit
