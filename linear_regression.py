
#  Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Helper print function for better visualization
def special_print(value):
    return print(f'------------------\n{value}')

# Pre-processing Data
# Loading the data
veriler = pd.read_csv('satislar.csv')

# Splitting the data into X and y
satislar = veriler[['Satislar']]
aylar = veriler[['Aylar']]

# Train test split yapıyoruz 
# X kolonlarımız aylar içinde y kolonumuz satislar içinde
X_train, X_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)


# =============================================================================
# # Veri üzerinde normalizing uyguluyoruz. Standard Deviaton ve Mean de uygulanmış oluyor.
# # Hem outliers önleme içinde güzel bir yol
# sc = preprocessing.StandardScaler()
# 
# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)
# y_train = sc.fit_transform(y_train)
# y_test = sc.fit_transform(y_test)
# =============================================================================


# Now I am calling the Linear regression class and fit (train) my model with our features (X_train)
lr = LinearRegression()
lr.fit(X_train, y_train)

# Let's predict
tahmin = lr.predict(X_test)


# Let's visualize our data. We are using matplotlib for this.
# But first we need to sort by index because we used random shuffling for train_test_split

X_train = X_train.sort_index()
y_train = y_train.sort_index()

plt.plot(X_train, y_train)

# Aşağıdaki değerleri plot edince aslında bir doğru çıktığını göreceğiz (Linear line) 
# Bunun sebebi predict ettiğinde aslında bu doğru çizgiyi (linear'ı) predict etmesi. 
# O yüzden Linear regression modeli bu
plt.plot(X_test, lr.predict(X_test))
plt.show()



