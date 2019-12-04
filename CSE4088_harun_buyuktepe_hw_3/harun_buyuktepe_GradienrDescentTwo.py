#harun_buyuktepe_
from numpy import *

iterations = 0
tolerance = 1e-14
learningRate = 0.1

u_now = 1.0
v_now = 1.0

ErrorFunction = lambda u, v: (u * exp(v) - 2 * v * exp(-u)) ** 2  #Error function defined here
DerivativeOfU = lambda u, v: (2 * (exp(v) + 2 * v * exp(-u)) * (u * exp(v) - 2 * v * exp(-u)))  #Gradient function of u
DerivativeOfV = lambda u, v: (2 * (u * exp(v) - 2 * exp(-u)) * (u * exp(v) - 2 * v * exp(-u)))  #Gradient function of u
ErrorInTime = ErrorFunction(u_now, v_now)  # initialize error function


while ErrorInTime >= tolerance:  # while error is greater than tolerance
    print("Error rate {} at iteration ".format(ErrorInTime),iterations)
    u_next = u_now - learningRate * DerivativeOfU(u_now, v_now)  #compute u at t + 1
    v_next = v_now - learningRate * DerivativeOfV(u_now, v_now)  #compute v at t + 1

    v_now = v_next  #v decided v_t+1
    u_now = u_next  # u decided u_t+1

    ErrorInTime = ErrorFunction(u_next, v_next)  #compute new error

    iterations +=1

print("Error rate {} at iteration ".format(ErrorInTime),iterations)
print("{} iterations were be needed.".format(iterations))
print("U value is ",u_now," and V value is ",v_now," at iteration ",iterations)