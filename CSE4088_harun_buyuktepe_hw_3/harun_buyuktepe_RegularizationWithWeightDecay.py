#harun_buyuktepe_
import re
from numpy import *

def readFile(fname):
    d = []
    with open(fname) as f:
        for l in f:
            l = l.strip()
            tok = map(lambda x: float(x), re.split('\s+', l))
            d.append(tok)
    return (d)

def getdata(fname='in.dta.txt'):
    d=genfromtxt(fname, dtype='float')
    x = array([i[0:2] for i in d])
    y = array([i[2] for i in d])
    return (x, y)

def lm_ridge(X, y, kValue):
    Xt = transpose(X)
    k = X.shape[1]
    lambdaeye = kValue * eye(k)
    m = matrix(dot(Xt, X) + lambdaeye) #dot product
    matrixInverse = m.getI() #Inverse of matrix
    beta = dot(dot(matrixInverse, Xt), y)
    return (beta.getA()[0, :])


def nonLinearTransform(x): #predefined nonLinearTransform function
    x1 = x[0];
    x2 = x[1]
    return ([1, x1, x2, x1 ** 2, x2 ** 2, x1 * x2, abs(x1 - x2), abs(x1 + x2)])


def makeClasification(x, w): #predefined operation
    z = nonLinearTransform(x)
    y = dot(z, w)
    return (1 if y >= 0 else -1)


def getError(X, y, w):
    n = X.shape[0]
    calculatedY = apply_along_axis(makeClasification, 1, X, w)
    errorCount = 1.0 * sum(y * calculatedY < 0)
    return (errorCount / n)

def runlm(kValue=0):
    (xin, yin) = getdata('in.dta.txt')
    (xout, yout) = getdata('out.dta.txt')
    zin = apply_along_axis(nonLinearTransform, 1, xin)
    w = lm_ridge(zin, yin, kValue)
    Ein = getError(xin, yin, w)
    Eout = getError(xout, yout, w)
    print("W ",w)
    print("Ein ",Ein)
    print("Eout",Eout)
    return ({'w': w, 'Ein': Ein, 'Eout': Eout})

def q2():
    return (runlm(kValue=0))

def q3():
    kValue = 10 ** -3
    return (runlm(kValue=kValue))

def q4():
    kValue = 10 ** 3
    print(kValue)

    return (runlm(kValue=kValue))

def q5():
    ks = [2, 1, 0, -1, 2]
    lams = [10 ** k for k in ks]
    Eouts = map(lambda x: runlm(kValue=x)['Eout'], lams)
    next(Eouts)  # prints 1
    return (Eouts)

def q6():
    ks = range(-6, 6)
    lams = [10 ** k for k in ks]
    Eouts = map(lambda x: runlm(kValue=x)['Eout'], lams)
    next(Eouts)  # prints 1
    return (Eouts)

print("\nQuestion 2")
runlm(0)

print("\nQuestion 3")
kValue = 10 ** -3
runlm(kValue)

print("\nQuestion 4")
kValue = 10 ** 3
runlm(kValue)

print("\nQuestion 5")
ks = [2, 1, 0, -1, 2]
lams = [10 ** k for k in ks]
Eouts = map(lambda x: runlm(x)['Eout'], lams)
print("\n2.")
next(Eouts)
print("\n1.")
next(Eouts)
print("\n0.")
next(Eouts)
print("\n-1.")
next(Eouts)
print("\n-2.\n")
next(Eouts)



