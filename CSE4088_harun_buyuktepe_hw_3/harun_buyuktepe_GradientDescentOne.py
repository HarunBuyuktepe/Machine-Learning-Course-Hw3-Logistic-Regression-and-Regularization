#harun_buyuktepe_
from sympy import *

u, v = symbols('u v')
expr = ((exp(u)*v) - 2*(v*exp(-u)))**2

print("Expression : {} ".format(expr))

expr_diff = Derivative(expr, u)

print("Derivative of expression with respect to u : {}".format(expr_diff))
print("Value of the derivative : {} ".format(expr_diff.doit())) 