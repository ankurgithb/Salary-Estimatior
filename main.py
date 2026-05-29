import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

df = pd.read_csv("Salary_data.csv")
# print(df.isnull().sum())
df = df.dropna()

# print(df.head()) 
# print(df.tail())
# print(df.describe())
# print(df.info())


df_encd = pd.get_dummies(df, columns=["Gender", "Education Level"], dtype = int)
# print(df_encd.head())
# print(df)

# correlation = df_encd.select_dtypes(include = np.number).corr()

# correlation_salary = correlation["Salary"].sort_values(ascending = False)
# print(correlation_salary)

X = df_encd.drop(["Salary", "Job Title"], axis = 1)
Y = df_encd["Salary"]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 11)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

# table = pd.DataFrame({"Actual" : Y_test , "predicted" : Y_pred})
# print(table)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("MAE = ",mae)
print("MSE = ",mse)
print('R2_VALUE = ',r2)
print('RMSE = ',rmse)

plt.scatter(range(len(Y_test)), Y_test, label = "Actual salary", color = "red")
plt.scatter(range(len(Y_pred)), Y_pred, label = "Predicted salary", color = "blue")
# plt.plot(Y_test.values, label="Actual")
# plt.plot(Y_pred, label="Predicted")
plt.xlabel("Index")
plt.ylabel("Salary")
plt.legend()
plt.grid(alpha = 0.4)
# plt.savefig("graph.png")
plt.savefig("graph_scatter.png")
plt.show()


# User input :-

age = int(input("Age :"))
experience = float(input("Years of Experience :"))
gender = input("Gender (Male/Female) :")
education = input("Education level :")

user_data = {
    "Age": age,
    "Years of Experience": experience,
    "Gender_Female": 1 if gender.lower() == "female" else 0,
    "Gender_Male": 1 if gender.lower() == "male" else 0,
    "Education Level_Bachelor's": 1 if education.lower() == "bachelor's" else 0,
    "Education Level_Master's": 1 if education.lower() == "master's" else 0,
    "Education Level_PhD": 1 if education.lower() == "phd" else 0
}

user_df = pd.DataFrame([user_data])

user_df = user_df.reindex(columns = X.columns, fill_value = 0)

predicted_salary = model.predict(user_df)
print("Estimated Salary =", predicted_salary[0])
