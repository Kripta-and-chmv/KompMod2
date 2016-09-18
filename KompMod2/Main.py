import Functions

test1 = Functions.Generator()

while True:

    test1.Reading('input.txt')
    test1.GenerateSequence(1000, 1)
    sequence=test1.GetSequence()

    period = Functions.Tests.FindPeriod(sequence)
    print(period)
    if (period<100):
        Functions.RandomCoeff('input.txt')
    else:
        break


intervals = Functions.Tests.Test_2(sequence, test1.m)
print (intervals)