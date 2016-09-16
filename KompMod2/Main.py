import Functions

test1 = Functions.Generator()
test1.Reading('input.txt')
test1.GenerateSequence(100, 1)

k=test1.GetSequence()

print(k)

period = Functions.Tests.FindPeriod([1, 8, 4, 0, 12, 8, 0, 8, 12, 16, 16, 0, 12, 8, 0, 8, 12, 16, 16, 0, 12, 8, 0, 8, 12, 16, 16, 0, 12, 8, 0, 8, 12, 16, 16, 0, 12, 8, 0, 8, 12, 16, 16, 0, 12, 8, 0, 8, 12, 16, 16, 0, 12, 8, 0, 8, 12, 16, 16, 0, 12, 8, 0, 8, 12, 16, 16, 0, 12, 8, 0, 8, 12, 16, 16, 0, 12, 8, 0, 8, 12])
print(period)