import numpy as np
import matplotlib.pyplot as plt
from regression import Regression
 
def main():
    #read input data
    my_data = np.genfromtxt('regression_data.txt', dtype=float, delimiter=',')
     
    # create input and output numpy arrays     
    x = my_data[:,0]
    y = my_data[:,1]
     
    # create regression class object
    reg = Regression()
     
    # set learning rate
    reg.set_learning_rate(0.001)
     
    # set maximum iterations
    reg.set_max_iterations(20000)
     
    # set l1 and l2 penalty
    reg.set_l1_penalty(0.1)
    reg.set_l2_penalty(0.1)
     
    # set tolerance
    reg.set_tolerance(1e-5)
     
    # fit a polynomial regression model 
    theta, cost, it = reg.polynomial_regression(x, y, 6)
     
    print "Regression coefficients :" + str(theta)
    print "Minimum cost function: " + str(cost)
    print "Iterations taken: " + str(it)
     
    # predict values for new input
    z = np.linspace(-2, 2, 4/0.01)
    prediction = reg.predict(z)
     
    # plot
    fig = plt.figure()
    plt.plot(x,y,'.', label='Input data')
    plt.plot(z,prediction,'r-', label='Prediction')
    plt.legend(loc=4)
    fig.suptitle('Polynomial Regression Fit')
    plt.xlabel('x (input)')
    plt.ylabel('y (predicted)')
    plt.savefig('fit_values.eps')
    plt.show()
 
 
if __name__ == "__main__":
    main()
