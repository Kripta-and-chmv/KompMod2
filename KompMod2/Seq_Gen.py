class SeqGenerator(object):

    def __init__(self):
        self.__a = 0
        self.__b = 0
        self.__c = 0
        self.__mod = 0
        self.__sequence = []

    def read_file(self, fileName):
        koef = []
        with open(fileName, 'r') as f:
            fileStr = f.readline()
            koef = fileStr.split(' ')
        for i in range(len(koef)):
            koef[i] = int(koef[i])
        self.__a, self.__b, self.__c, self.__mod = koef[0], koef[1], koef[2], koef[3]

    def generate_sequence(self, length, x0):

        self.__sequence = []
        self.__sequence.append(0)
        self.__sequence.append(0)
        self.__sequence.append(x0)

        for i in range(2, length + 2):
            self.__sequence.append(
                (self.__a * self.__sequence[i] + self.__b * self.__sequence[i - 2] +
                    self.__c) % self.__mod
            )
        self.__sequence = self.__sequence[3:]

    def get_sequence(self):
        return self.__sequence

    def get_mod(self):
        return self.__mod