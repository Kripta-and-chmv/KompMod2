import math


def Reading(fileName):
    koef=[]
    with open(fileName, 'r') as f:
        fileStr=f.readline()
        koef=fileStr.split(' ')
    for i in range(len(koef)):
        koef[i]=int(koef[i])
    
    return koef

def GenerateSequence(length, koef):
    sequence=[]
    a, b, c, m = koef[0], koef[1], koef[2], koef[3]
    Reading('input.txt')
    x0=0
    xnext=0
    xn=x0
    xnprev=0
    xnprevprev=0
    for i in range(length):
        xnext=(a*xn+b*xnprevprev+c) % m
        sequence.append(xnext)
        xprevprev=xnprev
        xnprev=xn
        xn=xnext
    return sequence
        