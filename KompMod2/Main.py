import Functions

test1 = Functions.Generator()
test1.Reading('input.txt')
test1.GenerateSequence(130, 1)

k=test1.GetSequence()

print(k)

period = Functions.Tests.FindPeriod(test1, 1)
print(period)