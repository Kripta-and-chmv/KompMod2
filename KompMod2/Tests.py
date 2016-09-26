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
                    # добавляем единицу,
                    # т.к. i указывает на начало второго периода
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

    def writing_in_file(wfile, length, hit, interval):
        """В файл wfile записывается размер выборки length, успешность
        прохождения теста hit, мат ожидание length/2 и доверительный интервал
        interval."""
        wfile.write(
            '============================== Тест 1 '
            '==============================\n\n')
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

    writing_in_file(wfile, len(sequence), hit, interval)

    return hit


def test_2(seq, mod, alpha, intervals_amount, drawing_graph, wfile):
    """Подаётся последоватеьность seq, mod - размерность алфавита,
        alpha - уровень значимости,
        intervalAmount - количество создаваемых интервалов,
        drawing_graph - bool значение, указывающее,
        нужно ли рисовать гистограмму,
        wfile - файл, куда будет записываться результат выполнения теста.
        возвращается значение bool - результат попадания
        теоретических мат. ожиданий и дисперсий
        в построенные доверительные интервалы
    """
    def create_intervals(mod, intervals_amount):
        """разбиваем отрезок от 0 до mod на amount интервалов.

        возвращается массив со значениями всех границ интервалов

        """
        intervals = []
        intervals.append(0)
        inter_length = mod / intervals_amount
        last_point = inter_length
        for i in range(intervals_amount - 1):
            intervals.append(last_point)
            last_point += inter_length
        intervals.append(mod)
        return intervals

    def find_frequency(intervals, sequence):
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

    def draw_histogram(frequency, intervals):
        """Рисует гистограмму частот frequency на интервалах intervals."""

        # ширина стобца - размер алфавита делаится на количество интервалов
        width = intervals[len(intervals) - 1] / (len(intervals) - 1)

        plt.bar(intervals[:len(intervals) - 1], frequency, width)
        plt.title('Frecuency Histogram')
        plt.xlabel('intervals')
        plt.ylabel('relative frequency')
        plt.xticks(intervals)
        plt.show()

    def calculate_exp_value(sequence):
        """Возвращает мат ожидание выборки sequence."""
        exp_value = 0.0
        for i in range(len(sequence)):
            exp_value += sequence[i]
        exp_value /= len(sequence)
        return exp_value

    def calculate_variance(sequence, exp_value):
        """возвращает дисперсию выборки sequence с мат ожиданием exp_value."""
        variance = 0.0
        for i in range(len(sequence)):
            variance += (sequence[i] - exp_value) ** 2
        variance /= len(sequence) - 1
        return variance

    def evaluate_freq(freq, seq_length, U, intervals_amount):
        """принимает список частот freq, длину последвательности seq_length,
        квантиль U, кол-во интервалов intervals_amount.

        Возвращает частоту интервалы для каждой частоты, кол-во попавших
        в построенные ассимптотические доверительные интервалы теоретич.
        частот

        """

        def calculate_freq_intervals(freq, seq_length, intervals_amount):
            """возвращает список ассимптотических доверительных интервалов для
            каждого значения из списка частот freq.

            каждый i-ый элемент freq_intervals[0] - няжняя граница дов интервала для freq[i],
            каждый i-ый элемент freq_intervals[1] - верхняя граница дов интервала для freq[i]

            """
            freq_intervals = [[], []]
            for i in range(len(freq)):
                delta = (U / intervals_amount) * \
                    math.sqrt((intervals_amount - 1) / seq_length)
                freq_intervals[0].append(freq[i] - delta)
                freq_intervals[1].append(freq[i] + delta)
            return freq_intervals

        # количество частот, попавших в доверительные интервалы
        hits_amount = 0

        intervals = calculate_freq_intervals(freq, seq_length,
            intervals_amount)
        P = 1 / intervals_amount
        for i in range(len(freq)):
            if(intervals[0][i] <= P <= intervals[1][i]):
                hits_amount += 1

        return P, intervals, hits_amount

    def evaluate_exp_value(exp_value, variance, seq_length, mod, U):
        """Принимает мат ожидание exp_value, дисперсию variance, длину выборки
        seq_length, размер алфавита mod, квантиль U. Возвращает теоретическое
        мат ожидание, интервал для построенного мат ожидания,

        bool значение - результат попадания теоретического мат ожидания в построенныей ассимптотический доверительный интервал

        """
        def calculate_exp_val_interval(exp_value, variance, seq_length, U):
            """Возвращает ассимптотический доверительный интервал для мат
            ожидания exp_value.

            exp_val_interval[0] - нижняя граница. exp_val_interval[1] - верхняя граница

            """
            exp_val_interval = []
            delta = U * math.sqrt(variance) / math.sqrt(seq_length)
            exp_val_interval.append(exp_value - delta)
            exp_val_interval.append(exp_value + delta)
            return exp_val_interval

        Mk = mod / 2
        exp_value_interval = calculate_exp_val_interval(
            exp_value, variance, seq_length, U)
        hit = False
        if(exp_value_interval[0] <= Mk <= exp_value_interval[1]):
            hit = True

        return Mk, exp_value_interval, hit

    def evaluate_variance(variance, seq_length, alpha, mod):
        """принимает дисперсию variance, длину выборки sqeLength, уровень
        значимости alpha, размер алфавита mod. Возвращает теоретическую
        дисперсию, интервал для построенной дисперсии,

        bool значение - результат попадания теоретической дисперсии в построенныей ассимптотический доверительный интервал

        """
        def calculate_variance_interval(variance, seq_length, alpha):
            """Возвращает ассимптотический доверительный интервал для дисперсии
            variance.

            variancInterval[0] - нижняя граница. variancInterval[1] - верхняя граница.

            """
            variance_interval = []
            variance_interval.append(
                (seq_length -
                 1) *
                variance /
                scipy.stats.chi2.ppf(
                    1 -
                    alpha /
                    2,
                    seq_length -
                    1))
            variance_interval.append(
                (seq_length -
                 1) *
                variance /
                scipy.stats.chi2.ppf(
                    alpha /
                    2,
                    seq_length -
                    1))
            return variance_interval

        Dk = mod ** 2 / 12
        variance_interval = calculate_variance_interval(variance,
                                                        seq_length, alpha)
        hit = False
        if(variance_interval[0] <= Dk <= variance_interval[1]):
            hit = True

        return Dk, intervals, hit

    def writing_in_file(wfile, length, intervals_amount, hits_amount,
                        teor_freq, freq_interv, exp_v_hit,
                        teor_exp_v, exp_v_interval, var_hit,
                        teor_var, var_interval):
        """В файл wfile записывается размер выборки length, количество
        построенных интервалов, теоретическая вероятность попадания элемента в
        интервал, количество попавших в интервалы элементов, интервалы для
        каждой частоты попадания в интервал, результат попадания мат ожидания в
        интервал, теоретичесоке мат ожидание, доверительный интервал для мат
        ожидания, результат попадания дисперсии в инервал, теоретическая
        дисперсия, доверительный интервал для дисперсии."""
        wfile.write(
            '============================== Тест 2 '
            '==============================\n\n')
        wfile.write('Размер выборки: %s\n' % (length))
        wfile.write('Количество интервалов: %s\n' % (intervals_amount))
        wfile.write(
            'Количество попаданий теор частоты в интервалы: %s\n' %
            (hits_amount))
        wfile.write('Теоретическая частота попаданий: %s\n' % (teor_freq))
        wfile.write(
            'Доверительный интервалы для каждой вычесленной частоты: \n')
        for i in range(intervals_amount):
            wfile.write('\t[%s, %s]\n' % 
                (freq_interv[0][i], freq_interv[1][i]))

        wfile.write(
            'Попадание теоретич. математическое ожидание '
            'в доверительный интервал: %s\n' %
            (exp_v_hit))
        wfile.write('Теоретическое математическое ожидание: %s\n' %
            (teor_exp_v))
        wfile.write(
            'Доверительный интервал для мат ожидания: [%s, %s]\n\n' %
            (exp_v_interval[0], exp_v_interval[1]))

        wfile.write(
            'Попадание теоретич. дисперсии в доверительный интервал: %s\n' %
            (var_hit))
        wfile.write('Теоретическая дисперсия: %s\n' % (teor_var))
        wfile.write(
            'Доверительный интервал для дисперсии: [%s, %s]\n\n' %
            (var_interval[0], var_interval[1]))

    # вычисляем квантиль уровня 1 - alpha/2 нормального распределения с мат.
    # ожид. 0 и среднеквадратич. отклонением 1
    U = scipy.stats.norm.ppf(1 - alpha / 2)
    # разбиваем на intervals_amount интервалов

    intervals = create_intervals(mod, intervals_amount)

    length = len(seq)

    freq = find_frequency(intervals, seq)
    if (drawing_graph is True):
        draw_histogram(freq, intervals)
    exp_value = calculate_exp_value(seq)
    variance = calculate_variance(seq, exp_value)

    teor_freq, freq_interv, hits_amount = evaluate_freq(
        freq, length, U, intervals_amount)
    teor_exp_v, exp_v_interval, exp_v_hit = evaluate_exp_value(
        exp_value, variance, length, mod, U)
    teor_var, var_interval, var_hit = evaluate_variance(
        variance, length, alpha, mod)

    writing_in_file(
        wfile,
        length,
        intervals_amount,
        hits_amount,
        teor_freq,
        freq_interv,
        exp_v_hit,
        teor_exp_v,
        exp_v_interval,
        var_hit,
        teor_var,
        var_interval)

    return exp_v_hit and var_hit


def test_3(seq, mod, alpha, subseqs_amount, intervals_amount, wfile):
    """Принимает выборку seq, размер алфавита mod, уровень значимости alpha,
    количество подпоследовательностей subseqAmount, количество интервалов для
    второго теста intervals_amount Возвращает False, если хотя бы одна
    подпоследовательность не прошла Test 1 и Test 2, иначе True."""

    def create_subseqs(seq, subseqs_amount):
        """принимает выборку seq и кол-во подпоследовательностей subseqs_amount.

        возвращает список подпоследовательностей, содержащий все
        элементы из seq

        """
        seq_length = len(seq)
        t = int((seq_length - 1) / subseqs_amount)
        seq_array = []

        for i in range(subseqs_amount):
            s = []
            for j in range(t + 1):
                s.append(seq[j * subseqs_amount + i])
            seq_array.append(s)
        return seq_array

    seq_array = create_subseqs(seq, subseqs_amount)
    is_test1 = True
    is_test2 = True

    wfile.write(
        '============================== Тест 3 '
        '==============================\n\n')
    wfile.write('Размер выборки: %s\n' % (seq))
    wfile.write('Количество подпоследовательностей: %s\n\n' % (subseqs_amount))
    for i in range(len(seq_array)):
        wfile.write(
            '\t\tПрохождение тестов 1 и 2 для  %s подпоследовательности\n\n' %
            (i + 1))
        hit = test_1(seq_array[i], alpha, wfile)
        if (hit is False):
            is_test1 = False

        hit = test_2(seq_array[i], mod, alpha, intervals_amount, False, wfile)
        if (hit is False):
            is_test2 = False

    return is_test1 and is_test2


def anderson_darling_test(sequence_, mod, wfile):
    """Тест андерсона на равномерное распределение. Принимает
    последовательность и размер алфавита.

    Возвращает bool значение - результат прохождения теста

    """
    def uniform_distr_func(x, mod):
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

    def writing_in_file(wfile, length, S, Scrit, hit):
        """В файл wfile записывается размер выборки length, значение статистики
        и критическое значение."""
        wfile.write(
            '============================== Тест Андерсона '
            '==============================\n\n')
        wfile.write('Размер выборки: %s\n' % (length))
        wfile.write('Успешность прохождения: %s\n' % (hit))
        wfile.write('Значение статистики: %s\n' % (S))
        wfile.write('Критическое значение: %s\n\n' % (Scrit))

    sequence = []
    sequence = sorted(copy.copy(sequence_))

    addition = 0
    length = len(sequence)

    for i in range(1, length + 1):
        F = uniform_distr_func(sequence[i - 1], mod)
        if(F == 0):
            return False, [999999999999, 2.4924]
        addition += (2 * i - 1) * math.log(F) / (2 * length)
        addition += (1 - (2 * i - 1) / (2 * length)) * math.log(1 - F)

    S = -len(sequence) - 2 * addition
    critical_value = 2.4924

    hit = False

    if (S <= critical_value):
        hit = True

    writing_in_file(wfile, len(sequence), S, critical_value, hit)

    return hit


def chisqr_test(sequence, mod, alpha, intervals_amount, wfile):
    """Нв вход подаются выюорка sequence, размерность алфавита mod.

    Возвращается значение bool - результат прохождения теста

    """
    def create_intervals(mod, intervals_amount):
        """разбиваем отрезок от 0 до mod на amount интервалов.

        возвращается массив со значениями всех границ интервалов

        """
        intervals = []
        intervals.append(0)
        inter_length = mod / intervals_amount
        last_point = inter_length
        for i in range(intervals_amount - 1):
            intervals.append(last_point)
            last_point += inter_length
        intervals.append(mod)
        return intervals

    def calculate_hits_amount(intervals, sequence):
        """находит количество элементов sequence, попадающих в каждые интервалы
        intervals."""
        frequency = numpy.zeros(len(intervals) - 1)
        length = len(sequence)
        for i in range(length):
            for j in range(len(intervals) - 1):
                if (intervals[j] <= sequence[i] < intervals[j + 1]):
                    frequency[j] += 1

        return frequency

    def calculate_probability_intervals(intervals, a, b):
        """находим вероятность попадания сл величины в интервалы intervals при
        равномерном распределении. возвращаем список вероятностей.

        a - нижняя граница равномерного распределения, b - верхняя

        """
        probabil = []
        for i in range(len(intervals) - 1):
            probabil.append((intervals[i + 1] - intervals[i]) / (b - a))
        return probabil

    def draw_histogram(frequency, intervals):
        """Рисует гистограмму частот frequency на интервалах intervals."""

        # ширина стобца - размер алфавита делаится на количество интервалов
        width = intervals[len(intervals) - 1] / (len(intervals) - 1)

        plt.bar(intervals[:len(intervals) - 1], frequency, width)
        plt.title('Chi2 Histogram')
        plt.xlabel('intervals')
        plt.ylabel('hits amount')
        plt.xticks(intervals)
        plt.show()

    def writing_in_file(wfile, length, S, Scrit, hit, interv_amount):
        """В файл wfile записывается размер выборки length, значение статистики
        и критическое значение."""
        wfile.write(
            '============================== Тест Андерсона '
            '==============================\n\n')
        wfile.write('Количество интервалов: %s\n\n' % (interv_amount))
        wfile.write('Успешность прохождения: %s\n' % (hit))
        wfile.write('Значение статистики: %s\n' % (S))
        wfile.write('Критическое значение: %s\n\n' % (Scrit))

    intervals = create_intervals(mod, intervals_amount)
    hits_amount = calculate_hits_amount(intervals, sequence)

    probabil = calculate_probability_intervals(intervals, 0, mod)

    draw_histogram(hits_amount, intervals)

    # вычисляется статистика
    addition = 0
    for i in range(intervals_amount):
        addition += (hits_amount[i] / len(sequence) -
                     probabil[i]) ** 2 / probabil[i]
    S = len(sequence) * addition

    r = 5
    def integrand(x, r):
        return x ** (r / 2 - 1) * sympy.exp(-x / 2)

    Schi2 = scipy.integrate.quad(integrand, S, numpy.inf, args = (r))
    k =  2 ** (r / 2) * math.gamma(int(r / 2))
    k = Schi2[0] / k

    hit_chi2 = False
    if (k > alpha):
       hit_chi2 = True

    S_crit = 18.307
    hit = False
    if(S <= S_crit):
        hit = True

    writing_in_file(wfile, len(sequence), S, S_crit, hit, intervals_amount)

    return hit
