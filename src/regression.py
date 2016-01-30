import numpy as np

class Regression:
    """
    Class Regression : fits a linear regression model to data
    """
    
    def __init__(self):
        self.max_iter = 1000
        self.learning_rate = 0.01
        self.order = 1
        self.theta = np.zeros(2)
        self.l1_penalty = 0
        self.l2_penalty = 0
        self.tolerance = 1e-4

    def set_max_iterations(self, iterations):
        """
        this function sets max iterations for the optimizer
        @param iterations: max iterations
        """
        self.max_iter = iterations
    
    def set_learning_rate(self, alpha):
        """
        this function sets learning rate for the optimizer
        @param alpha: learning rate
        """
        self.learning_rate = float(alpha)
    
    def set_l2_penalty(self, l2_coeff):
        """
        this function sets l2 penalty 
        @param l2_coeff: l2_penalty coefficient
        """
        self.l2_penalty = float(l2_coeff)
        
    def set_l1_penalty(self, l1_coeff):
        """
        this function sets l1 penalty 
        @param l2_coeff: l1_penalty coefficient
        """
        self.l1_penalty = float(l1_coeff)
    
    def set_tolerance(self, tol):
        """
        this function sets tolerance.
        If improvement in successive iterations is less than tolerance, solver exits
         
        @param tol: tolerance  
        """
        self.tolerance = float(tol)
        
    def polynomial_regression(self, x, y, deg):
        """
        this function fits a polynomial regression to given input
        @param x: input vector
        @param y: target vector
        @param deg: polynomial degree
        
        @return: theta: regression coeficient
        @return: const_fn: const_function evaluations 
        
        This function evaluates: 
            minimize(x,y,theta) = 1/(2*n)(sum(theta*x-y)^2 + sum(l1*theta) + sum(l2*theta^2))
            using gradient descent optimization method
        """
        
        # raise exception if polynomial degree is less than 0
        if (deg < 0):
            raise ValueError('Polynomial Degree must be > 0')
        
        # also account for intercept
        self.order = deg+1
        features = np.empty([len(x), self.order], dtype=float)
        self.theta = np.zeros(self.order)
        cost_prev = 0
        
        # populate feature vector: feature[i] = [1, x[i], x[i]^2,...x[i]^deg]
        for i in range(len(x)):
            for p in range(self.order):
                features[i][p] = pow(x[i], p)
    
        log = 0
        # repeat for max_iter times
        for repeat in range(self.max_iter):
            
            # calculate sum(theta.transpose().x - y)
            mat_mult = (features.dot(self.theta) - y)
            
            # we do not regularize theta[0]
            self.theta[0] -= (self.learning_rate/len(x))*(mat_mult*features[:,0]).sum()
            
            # update theta using regularization, considering l1 and l2 penalty
            for j in range(1, self.order):
                self.theta[j] -= (self.learning_rate/len(x))*((mat_mult*features[:,j]).sum() 
                                                              + self.l1_penalty + self.l2_penalty*self.theta[j])
            
            # calculate cost function after every 10 iterations
            if(repeat%10 == 0):
                cost = (1./(2*len(x))) * (np.power(features.dot(self.theta) - y, 2).sum() 
                                                  + self.l1_penalty*self.theta.sum() 
                                                  + self.l2_penalty*np.power(self.theta,2).sum())
                
                # if change in cost function < tolerance, break the loop
                if(log > 1 and np.abs(cost - cost_prev) < self.tolerance):
                    break
                log += 1
                cost_prev = cost
    
        # return theta and const_fn
        return self.theta, cost, repeat
    
    def linear_regression(self,x,y):
        """
        fit a linear regression to input data
        @param x: input vector
        @param y: target vector
        """
        return self.polynomial_regression(x, y, 1)
    
    def predict(self,x):
        """
        predicts the output for input 
        @param x: input value/vector
        @return: predictions : theta*polynomial(x)
        """
        result = np.zeros(len(x))
        for i in range(self.order):
            result += self.theta[i]*np.power(x,i)
        return result
    
