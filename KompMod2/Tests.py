import math
import numpy
import copy
import random
import sympy
import scipy
import scipy.stats
import matplotlib.pyplot as plt


def find_period(sequence):
    """Находит период у числовой последовательности.
    Аргументы: sequence - последовательность.
    Вывод: period - длина периода.

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
    """Проходится тест №1. Результаты записываются в файл.
    Аргументы:
        sequence - выборка, list числовых значений;
        alpha - уроень значимости, float;
        wfile - файл, куда записываются результаты теста, file.

    Вывод: 
        hit - успешность прохождения, bool.
        

    """

    def writing_in_file(wfile, length, hit, interval):
        """Запись результатов в файл    
        Аргументы:
            wfile - файл, куда происходит запись, file;
            length - длина выборки, int;
            hit - успешность прохождения теста, bool;
            interval - доверительный интервал, list из двух элементов,
                [0] - начало интервала, [1] - конец.

        """
        teor_exp_value = length / 2
        wfile.write(
            '============================== Тест 1 '
            '==============================\n\n')
        wfile.write('Размер выборки: %s\n' % (length))
        wfile.write('Успешность прохождения: %s\n' % (hit))
        wfile.write('Математическо ожидание: %s\n' % (teor_exp_value))
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
    """Тест №2.
    Аргументы:
        seq - выборка, list числовых значений;
        mod - размерность алфавита выборки, int;
        alpha - уровень значимости, float;
        intervals_amount - количество интервалов, int;
        drawing_graph - нужно ли рисовать гистограмму, bool;
        wfile - файл, куда записываются результаты теста, file.
    Вывод:
    exp_v_hit and var_hit - успешность прохождения теста, попадание теор.
        мат. ожидания и дисперсии в построенные доверительные интервалы, 
        bool.
    
    """

    def create_intervals(mod, intervals_amount):
        """Разбивает отрезок от 0 до mod на интервалы.
        Аргументы:
            mod - верхняя граница отрезка, число;
            intervals_amount - количество интервалов, int.
        Вывод:
            intervals - список с границами интервалов, list числовых значений.
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
        """Вычисляется частоту попаданий элементов выборки в каждый интервал.
        Аргументы:
            intervals - список границ интервалов, list;
            sequence - выборка, list числовых значений.
        Вывод:
            frequency - список количества попаданий для каждого интервала, list of int.

        """
        frequency = numpy.zeros(len(intervals) - 1)
        length = len(sequence)
        piece = 1 / length
        for i in range(length):
            for j in range(len(intervals) - 1):
                if (intervals[j] <= sequence[i] < intervals[j + 1]):
                    frequency[j] += piece

        return frequency

    def draw_histogram(frequency, intervals):
        """Рисует гистограмму частот.
        Аргументы:
            frequency - частота попаданий в интервалы, int;
            intervals - списо границ интервалов, list.
        """

        # ширина стобца - размер алфавита делаится на количество интервалов
        width = intervals[len(intervals) - 1] / (len(intervals) - 1)

        plt.bar(intervals[:len(intervals) - 1], frequency, width)
        plt.title('Frecuency Histogram')
        plt.xlabel('intervals')
        plt.ylabel('relative frequency')
        plt.xticks(intervals)
        plt.show()

    def calculate_exp_value(sequence):
        """Вычиляет математическое ожидание выборки.
        Аргументы: 
            sequence - выборка, list числовых значений.
        Вывод:
            exp_value - математическое ожидание, float.
        
        """
        exp_value = 0.0
        for i in range(len(sequence)):
            exp_value += sequence[i]
        exp_value /= len(sequence)
        return exp_value

    def calculate_variance(sequence, exp_value):
        """Вычиляет дисперсию выборки.
        Аргументы: 
            sequence - выборка, list числовых значений.
        Вывод:
            variance - дисперсия, float.
        
        """
        variance = 0.0
        for i in range(len(sequence)):
            variance += (sequence[i] - exp_value) ** 2
        variance /= len(sequence) - 1
        return variance

    def evaluate_freq(freq, seq_length, U, intervals_amount):
        """Оценивание частоты попаданий эелементов выборки в интервалы.
        Аргументы:
            freq - частота попаданий для кжадого интервала, list of int;
            seq_length - длина выборки, int;
            U - квантиль, float;
            intervals_amount - количество интервалов.
        Вывод:
            P - теоретическая частота попаданий, float;
            intervals - список доверительных интервалов для каждой частоты. 
                элемент [0][i] - нижняя граница для частоты попадания
                в i-ый интервал, элемент [1][i] - верхняя граница, list of lists;
            hits_amount - количество попаданий P в доверительные интервалы, int.

        """

        def calculate_freq_intervals(freq, seq_length, intervals_amount):
            """Построение доверительных интервалов для частот.
            Аргументы:
                freq - частоты для каждого интервала, list;
                seq_length - длина последовательности, int;
                intervals_amount - количество интервалов, int;
            Вывод:
                freq_intervals - список доверительных интервалов для каждой частоты. 
                элемент [0][i] - нижняя граница для частоты попадания
                в i-ый интервал, элемент [1][i] - верхняя граница, list of lists.

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
        """Оценивание попадания математического ожидания в доверительный интервал.
        Аргументы:
            exp_value - математическое ожидание, float;
            variance - дисперсия, float;
            seq_length - длина выборки, int;
            mod - размерность алфавита выборки, int;
            U - квантиль, float.
        Вывод:
            Mk - теоретическое математическое ожидание, float;
            exp_value_interval - доверительный интервал для мат ожидания, list числовых значений;
            hit - попадание Mk в доверительный интервал, bool.

        """
        def calculate_exp_val_interval(exp_value, variance, seq_length, U):
            """Вычисляется доверительный интервал для математического ожидания.
            Аргументы:
                exp_value - математическое ожидание, float;
                variance - дисперсия, float;
                seq_length - длина выборки, int;
                U - квантиль, float.
            Вывод:
                exp_val_interval - доверительный интервал для математического
                ожидания; [0] - нижняя граница, [1] - верхняя,
                list числовых значений.

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
        """Оценивание попадания дисперсии в доверительный интервал.
        Аргументы:
            variance - дисперсия, float;
            seq_length - длина выборки, int;
            alpha - уровень значимости, float;
            mod - размерность алфавита выборки, int.
        Вывод:
            Dk - теоретическая дисперсия, float;
            exp_value_interval - доверительный интервал для мат ожидания, list числовых значений;
            hit - попадание Mk в доверительный интервал, bool.

        """
        def calculate_variance_interval(variance, seq_length, alpha):
            """Вычисляется доверительный интервал для дисперсии.
            Аргументы:
                variance - дисперсия, float;
                seq_length - длина выборки, int;
                alpha - уровень значимости, float.
            Вывод:
                variance_interval - доверительный интервал для дисперсии. 
                    [0] - нижняя граница, [1] - верхняя, list числовых значений.

            """
            variance_interval = []
            variance_interval.append((seq_length - 1) * variance /
                scipy.stats.chi2.ppf(1 - alpha / 2, seq_length - 1))
            variance_interval.append((seq_length - 1) * variance /
                scipy.stats.chi2.ppf(alpha / 2, seq_length - 1))
            return variance_interval

        Dk = mod ** 2 / 12
        variance_interval = calculate_variance_interval(variance,
            seq_length, alpha)
        hit = False
        if(variance_interval[0] <= Dk <= variance_interval[1]):
            hit = True

        return Dk, variance_interval, hit

    def writing_in_file(wfile, length, intervals_amount, hits_amount,
                        teor_freq, freq_interv, exp_v_hit,
                        teor_exp_v, exp_v_interval, var_hit,
                        teor_var, var_interval):
        """Запись результатов в файл.
        Аргументы:
            wfile - файл, куда происходит запись, file;
            length - длина выборки, int;
            intervals_amount - количество интервалов, int;
            hits_amount - количество попаданий частот в доверит интервал, int;
            teor_freq - теоретическая частота попаданий, float;
            freq_interv - доверительные интеваы для каждой частоты, list of lists;
            exp_v_hit - попадание теор. мат. ожидания в доверительный интервал, bool;
            teor_exp_v - теоретическое мат. ожилание, float;
            exp_v_interval - доверительный интервал для мат ожидания, list;
            var_hit - попадание теор. дисперсии в доверительный интервал, bool;
            teor_var - теоретическая дисперсия, float;
            var_interval - доверительный интервал для дисперсии, list;

        """
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

    writing_in_file(wfile, length, intervals_amount, hits_amount, teor_freq,
        freq_interv, exp_v_hit, teor_exp_v, exp_v_interval, var_hit, teor_var,
        var_interval)
    
    return exp_v_hit and var_hit

def test_3(seq, mod, alpha, subseqs_amount, intervals_amount, wfile):
    """Проводится тест №3. Результаты записываются в файл.
    Аргументы:
        seq - выборка, list числовых значений;
        mod - размерность алфавита выборки, int;
        alpha - уровень значимости, foat;
        subseqs_amount - количество создаваемых подпоследовательностей, int;
        intervals_amount - количество интервалов для теста №2, int;
        wfile - файл, куда записываются результаты теста, file.
    Вывод:
        is_test1 and is_test2 - успешность прохождения всех подпоследовательностей
            тестов 1 и 2. В случае хотя бы одной неудачи - тест №3 неудачен, bool.

    """

    def create_subseqs(seq, subseqs_amount):
        """Создаёт подпоследовательности из заданной выборки.
        Аргументы:
            seq - выборка, list числовых значений;
            subseqs_amount - количество подпоследовательностей, int.
        Вывод:
            seq_array - список подпоследовательностей seq, list of lists.

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
    wfile.write('Размер выборки: %s\n' % (len(seq)))
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
    """Тест Андерсона-Дарлинга.
    Аргументы:
        sequence - выборка, list числовых значений;
        mod - размерность алфавита выборки, int;
        wfile - файл, куда записываются результаты теста, file.
    Вывод:
        hit and hit_a2 - успешность прохождения теста по двум критериям, bool.

    """
    def uniform_distr_func(x, a, b):
        """Функция равномерного распредленения.
        Аргументы:
            x - аргумент функции, числовое значение;
            a - нижняя граница функции распределения, int;
            b - верхняя граница функции распределения, int;
        Вывод:
            результат функции, чсловое значение.

        """
        if (x < a):
            return 0
        elif (a <= x < b):
            return (x - a) / (b - a)
        elif (x >= b):
            return 1

    def writing_in_file(wfile, length, S, Scrit, hit, a2, hit_a2):
        """Запись результатов в файл    
        Аргументы:
            wfile - файл, куда происходит запись, file;
            length - длина выборки, int;
            S - вычисленное значение статистики, float;
            Scrit - критическе значение статистики;
            hit - успешность прохождения теста, bool;
            a2 - значение a2, float;
            hit_a2 - прохождение теста по критерию а2, bool.
        """
        wfile.write(
            '============================== Тест Андерсона '
            '==============================\n\n')
        wfile.write('Размер выборки: %s\n' % (length))
        wfile.write('Успешность прохождения по критерию сравнения с крит. '
            'значением: %s\n' % (hit))
        wfile.write('Значение статистики: %s\n' % (S))
        wfile.write('Критическое значение: %s\n\n' % (Scrit))
        wfile.write('Успешность прохождения про критерию a2: %s\n\n' % (hit_a2))
        wfile.write('Значение а2: %s\n' % (a2))
        

    sequence = []
    sequence = sorted(copy.copy(sequence_))

    addition = 0
    length = len(sequence)

    for i in range(1, length + 1):
        F = uniform_distr_func(sequence[i - 1], 0, mod)
        if(F == 0):
            return False, [999999999999, 2.4924]
        addition += (2 * i - 1) * math.log(F) / (2 * length)
        addition += (1 - (2 * i - 1) / (2 * length)) * math.log(1 - F)

    S = -len(sequence) - 2 * addition

    def integrand(x, S, j):
        return sympy.exp(S/(8*(x**2 + 1)) - ((4*j + 1)**2 * math.pi**2 * x**2)/(8 * S))

    addition = 0
    for j in range(170):
        el = (-1)**j * (math.gamma(j + 0.5) * (4*j + 1)) / (math.gamma(0.5) * math.gamma(j + 1))
        el *= sympy.exp(-((4*j + 1)**2 * math.pi**2) / (8 * S))
        integ = scipy.integrate.quad(integrand, 0, numpy.inf, args = (S, j))
        el *= integ[0]
        addition += (math.sqrt(2 * math.pi) / S) * el
        
    a2 = 1 - addition
    hit_a2 = False
    if (a2 > 0.05):
        hit_a2 = True

    critical_value = 2.4924

    hit = False

    if (S <= critical_value):
        hit = True

    writing_in_file(wfile, len(sequence), S, critical_value, hit, a2, hit_a2)

    return hit and hit_a2

def chisqr_test(sequence, mod, alpha, intervals_amount, drawing_graph, wfile):
    """Тест Хи-квадрат.
    Аргументы:
        sequence - выборка, list числовых значений;
        mod - размерность алфавита выборки, int;
        alpha - уровень значимости, float;
        intervals_amount - количество интервалов, int;
        drawing_graph - нужно ли рисовать гистограмму, bool;
        wfile - файл, куда записываются результаты теста, file.
    Вывод:
        hit and hit_a2 - успешность прохождения теста по двум критериям, bool.

    """
    def create_intervals(mod, intervals_amount):
        """Разбивает отрезок от 0 до mod на интервалы.
        Аргументы:
            mod - верхняя граница отрезка, число;
            intervals_amount - количество интервалов, int.
        Вывод:
            intervals - список с границами интервалов, list числовых значений.

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
        """Вычисляется количество элементов выборки, попавших в каждый интервал.
        Аргументы:
            intervals - список границ интервалов, list;
            sequence - выборка, list числовых значений.
        Вывод:
            frequency - список количества попаданий для каждого интервала, list of int.

        """
        frequency = numpy.zeros(len(intervals) - 1)
        length = len(sequence)
        for i in range(length):
            for j in range(len(intervals) - 1):
                if (intervals[j] <= sequence[i] < intervals[j + 1]):
                    frequency[j] += 1

        return frequency

    def calculate_probability_intervals(intervals, a, b):
        """Вычисляется вероятность попадания слчайной величины в заданные
        интервалы при равномерном распределении.
        
        Аргументы:
            intervals - список границ интервалов, list;
            a - нижняя граница функции равномерного распределения;
            b - верхняя граница функции равномерного распределения.
        Вывод:
            probabil - список вероятностей для каждого интервала, list of float.

        """
        probabil = []
        for i in range(len(intervals) - 1):
            probabil.append((intervals[i + 1] - intervals[i]) / (b - a))
        return probabil

    def draw_histogram(frequency, intervals):
        """Рисует гистограмму частот.
        Аргументы:
            frequency - частота попаданий в интервалы, int;
            intervals - списо границ интервалов, list.
        """

        # ширина стобца - размер алфавита делаится на количество интервалов
        width = intervals[len(intervals) - 1] / (len(intervals) - 1)

        plt.bar(intervals[:len(intervals) - 1], frequency, width)
        plt.title('Chi2 Histogram')
        plt.xlabel('intervals')
        plt.ylabel('hits amount')
        plt.xticks(intervals)
        plt.show()

    def writing_in_file(wfile, length, S, Scrit, hit, a2, hit_a2, interv_amount):
        """Запись результатов в файл    
        Аргументы:
            wfile - файл, куда происходит запись, file;
            length - длина выборки, int;
            S - вычисленное значение статистики, float;
            Scrit - критическе значение статистики;
            hit - успешность прохождения теста по критерию сравнения с крит. значением, bool;
            a2 - a2, float;
            hit_a2 - успешность прохождения теста по критерию a2, bool;
            interv_amount - количество интервалов, int.

        """
        wfile.write(
            '============================== Тест Андерсона '
            '==============================\n\n')
        wfile.write('Количество интервалов: %s\n\n' % (interv_amount))
        wfile.write('Успешность прохождения по Sкрит: %s\n' % (hit))
        wfile.write('Значение статистики: %s\n' % (S))
        wfile.write('Критическое значение: %s\n\n' % (Scrit))
        wfile.write('Успешность прохождения по a2: %s\n' % (hit_a2))
        wfile.write('Значение a2: %s\n' % (a2))

    intervals = create_intervals(mod, intervals_amount)
    hits_amount = calculate_hits_amount(intervals, sequence)

    probabil = calculate_probability_intervals(intervals, 0, mod)

    if(drawing_graph is True):
        draw_histogram(hits_amount, intervals)

    # вычисляется статистика
    addition = 0
    for i in range(intervals_amount):
        addition += (hits_amount[i] / len(sequence) -
                     probabil[i]) ** 2 / probabil[i]
    S = len(sequence) * addition

    # вычисляется a2
    r = 5
    def integrand(x, r):
        return x ** (r / 2 - 1) * sympy.exp(-x / 2)

    a2 = scipy.integrate.quad(integrand, S, numpy.inf, args = (r))
    a2 = a2[0] / 2 ** (r / 2) * math.gamma(int(r / 2))

    hit_a2 = False
    if (a2 > alpha):
       hit_a2 = True

    S_crit = 18.307
    hit = False
    if(S <= S_crit):
        hit = True

    writing_in_file(wfile, len(sequence), S, S_crit, hit, a2, hit_a2, intervals_amount)

    return hit and hit_a2
