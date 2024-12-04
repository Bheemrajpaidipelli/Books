

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

#************************************************************************************************************************

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\bheem\OneDrive\Desktop\ML\SSR RESSIONS\Position_Salaries.csv")


X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, 2].values
#Dependent variable we will specify the index 2

#our main goal is to predict if this employee is bluffing  by building machine learning model that is polynomial model

#************************************************************************************************************************


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(X, y) # linear regression 
#fit the lin_reg object to X & y. now our simple linear regression is ready 

#*************************************************************************************************************************

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 7) # we mentoine 2 degree 

X_poly = poly_reg.fit_transform(X)


poly_reg.fit(X_poly, y)


#*************************************************************************************************************************

lin_reg_2 = LinearRegression()
#we crate an 2nd object for same LinearRegression
lin_reg_2.fit(X_poly, y) # non linear 


#*************************************************************************************************************************


#lets starts the plotting by true observation 
plt.scatter(X, y, color = 'red')
#we are going to plot for actual value of X & y
plt.plot(X, lin_reg.predict(X), color = 'blue')
#now plot for the prediction line where x coordinate are predictin points & for y-cordinates predicted value which is lin_reg.predict (x)
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



#************************************************************************************************************************

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
#in y-coordinate we have to replace with lin_reg2 which we create for poly regression model
#X_poly is not defined cuz we already defined in above plot, so insted of X_poly we will define complete fit_trasnform code 
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]]) # slr

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
#This code show me that predicted salary of 6.5 level using poly reg model
#That means employee is True and we solved this by using polyregression model















