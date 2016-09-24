class SeqGenerator(object):

    def __init__(self):
        self.a = 0
        self.b = 0
        self.c = 0
        self.mod = 0
        self.sequence = []

    def Reading(self, fileName):
        koef = []
        with open(fileName, 'r') as f:
            fileStr = f.readline()
            koef = fileStr.split(' ')
        for i in range(len(koef)):
            koef[i] = int(koef[i])
        self.a, self.b, self.c, self.mod = koef[0], koef[1], koef[2], koef[3]

    def GenerateSequence(self, length, x0):

        self.sequence = []
        self.sequence.append(0)
        self.sequence.append(0)
        self.sequence.append(x0)

        for i in range(2, length + 2):
            self.sequence.append
            (
                (self.a * self.sequence[i] + self.b * self.sequence[i - 2] +
                    self.c) % self.mod
            )
        self.sequence = self.sequence[3:]

    def GetSequence(self):
        return self.sequence
