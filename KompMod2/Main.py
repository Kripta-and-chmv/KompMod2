import Functions

test1 = Functions.Generator()
test1.Reading('input.txt')
test1.GenerateSequence(31, 1)

k=test1.GetSequence()

print(k)

period = Functions.Tests.FindPeriod(k)
print(period)