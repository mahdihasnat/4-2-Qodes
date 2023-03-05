import csv
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

date=[]
value=[]

with open("./NOK.csv", 'r') as file:
  csvreader = csv.reader(file)
  
#   skipping the header
  next(csvreader)
  
  for row in csvreader:
    date.append(row[0])
    value.append(float(row[4]))

date=numpy.array(date)
value=numpy.array(value)

Xs=[]
Ys=[]

for i in range(5, len(value)):
    X = value[i-5:i]
    Y = value[i]
    # print(X, Y)
    Xs.append(X)
    Ys.append(Y)

Xs=numpy.array(Xs)
Ys=numpy.array(Ys)

X_train, X_test, y_train, y_test = train_test_split(Xs,
                                                    Ys, 
                                                    test_size=0.2,
                                                    random_state=3)

model1 = LinearRegression(n_jobs=-1)
model1.fit(X_train, y_train)

model2 = KNeighborsRegressor(n_neighbors=2)
model2.fit(X_train, y_train)

model3 = make_pipeline(PolynomialFeatures(3), Ridge())
model3.fit(X_train, y_train)

model4 = GradientBoostingRegressor(n_estimators=100, 
                                    learning_rate=0.1, 
                                    max_depth=1, 
                                    random_state=0, 
                                    loss='squared_error')
model4.fit(X_train, y_train)

model5 = ElasticNet(random_state=0)
model5.fit(X_train, y_train)


# Find MSE values

print("MSE values: ")
print("Linear Regression: ", mean_squared_error(y_test, model1.predict(X_test)))
print("KNN Regression: ", mean_squared_error(y_test, model2.predict(X_test)))
print("Polynomial Regression: ", mean_squared_error(y_test, model3.predict(X_test)))
print("GradientBoostingRegressor: ", mean_squared_error(y_test, model4.predict(X_test)))
print("ElasticNet Regression: ", mean_squared_error(y_test, model5.predict(X_test)))

# exit()

# take 5 values as input from user
input=input('Enter 5 values (space-separated): ')
# print(input)
input=input.split()

X_=[]
for i in input:
    X_.append(float(i))
X_=numpy.array(X_)

# predict the value
print("-----------------------------")
print("Predicted value: ")
print("Linear Regression: ", model1.predict([X_]))
print("KNN Regression: ", model2.predict([X_]))
print("Polynomial Regression: ", model3.predict([X_]))
print("GradientBoostingRegressor: ", model4.predict([X_]))
print("ElasticNet Regression: ", model5.predict([X_]))
