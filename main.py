import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('pop_full.csv')
print(df.head())

print(df.info())

x = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values.reshape(-1, 1)
model = LinearRegression().fit(x, y)
y_pred = model.predict([[2019]])
print(y_pred)
# plot
fig, ax = plt.subplots()

ax.scatter(x, y)
ax.plot(x,float(model.coef_[0])*x+float(model.intercept_))
plt.show()