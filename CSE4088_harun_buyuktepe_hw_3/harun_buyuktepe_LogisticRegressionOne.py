#harun_buyuktepe_
from numpy import *

print("Started...")
def random2DimensionalWeight():          #Generate 2 dimensional random weight function

    a = random.uniform(-1, 1, (1, 2))[0]  #Generate random point x,y
    b = random.uniform(-1, 1, (1, 2))[0]  #Generate random point x,y
    w1 = (b - a)[1] / (b - a)[0]
    w0 = a[1] - w1 * a[0]
    weightFunction = array((w0, w1))  #return random weight
    return weightFunction


def randomDecisionMaker(N, weights_fx):         #Create a Function Generates Random Points and Labels
    c_2D_to_3D = lambda w_2D: hstack((-w_2D, 1))       #To convert 3 dimention from 2 dimention format

    points = hstack((ones((N, 1)), random.uniform(-1, 1, (N, 2))))     #fill with 1s
    weightFunction = c_2D_to_3D(weights_fx)
    result = dot(points, weightFunction)                 #Apply dot product for all random point(100 point in this que.)
    final = where(exp(result) / (1 + exp(result)) > 0.5, 1, -1)         #Decide with logistic regression formula and
    final = expand_dims(final, 1)  # Expand Array b                         # determined treshold value
    return points, final, weightFunction


def LogisticRegressionGradientDescent(N): #Apply gradient descent to minimize cost function

    randomWeight = random2DimensionalWeight()      # Create a Rand. Func. f(x)
    points, results, randomWeight3Dimension = randomDecisionMaker(N, randomWeight)      #Create environment

    epsilon = 0.01
    learningRate = 0.01         #Learning Rate
    epoch = 0               #epoch Number

    Wnow = zeros(len(randomWeight3Dimension))        #Initialize g(x) weights = 0

    tolerance = 9999999      #Initialize High Tolerance 

    while tolerance > epsilon:  #use iterative gradient descent
        Wcurr = copy(Wnow)  # w(n) = w(t)

        for n in arange(N):
            gradient = -(results[n] * points[n]) / (1 + exp(results[n] * dot(Wcurr, points[n])))
            Wcurr -= learningRate * gradient     #stochastic gradient updated

        Wnext = Wcurr         #w(t+1) = w(N)

        tolerance = linalg.norm(Wnow - Wnext)       # || w(t) - w(t + 1) ||
        Wnow = Wnext  #w(t) = w(t + 1)
        epoch += 1          #Increment epoch Number

    Weight3D = Wnow

    return Weight3D, points, results, epoch


N =100                      #Repeat Experiment 100 Times
E_outs = zeros(N)
epochs = zeros(N)

for runs in range(N):
    Weight3D, X, Y, epoch = LogisticRegressionGradientDescent(N)
    E_outs[runs] = mean(log(1 + exp(-Y * expand_dims(dot(X, Weight3D), 1))))
    epochs[runs] = epoch

print("\nThe Average E_out is %.3f" % mean(E_outs))
print("The Average Epochs is %d" % mean(epochs))