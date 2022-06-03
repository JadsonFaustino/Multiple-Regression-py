import numpy as np
import matplotlib as plt

# class for Linear Regressions
class PolynomialRegression:

  # to create this class, you need to give x and y arrays
  # so it'll deduce B0 and B1 coefficients
  def __init__(self, x, y, degree=2):
    
    expoArr = [np.ones(len(x))]
    for expo in range(1, degree+1):
      expoArr.append(np.power(x,expo))
    
    self.x = np.array(expoArr).transpose()
    
    self.y = np.array([y]).transpose()

    x_t = self.x.transpose()
    a = np.dot(x_t, self.x)
    b = np.dot(x_t, self.y)
    c = np.linalg.inv(a)

    self.B = np.dot(c, b)
    self.B = self.B.transpose().tolist()[0]
    self.x = self.x.transpose().tolist()
    self.y = self.y.transpose().tolist()
  
  # predict a value of y with a x given based on linear regression
  def predict(self, x_value):
    if ( (type(x_value) is not int) and (type(x_value) is not float) ):
      raise Exception("A numeric value must be given")
    predicted_value = 0
    for i in range(0, len(self.B) ):
        predicted_value += (x_value ** i) * self.B[i]
    return predicted_value

  # return an array of y predicted values with all x given based on linear regression
  def predictAll(self):
    allPredicts = []
    for each in self.x[1]:
      each = float(each)
      allPredicts.append( self.predict(each) )
    return allPredicts

  def R2(self):
    y_ = self.predictAll()
    SQe = 0
    Syy = 0
    for i in range(0, len(self.y)):
      SQe += (self.y[i] - y_[i]) ** 2
      Syy += (self.y[i] - self.y_mean) ** 2
    
    return (1 - SQe/Syy)

  # compare the original values to predicted values i
  def compare(self):
    import matplotlib.pyplot as plt

    plt.plot(self.x[1], self.y[0], 'g-', self.x[1], self.predictAll(), 'r-')
    plt.legend(['Base', 'Predict'])
    plt.title("Polynomial Regression")


x = [0,1,2,3,4,5,6,7,8,9,10]
y = [2,5,6,7,9,8,5,2,6,4,8]
obj = PolynomialRegression(x,y,8)
obj.compare()
