import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#data = pd.read_csv('pokemon.csv')

##corelation map
#,ax = plt.subplots(figsize=(18, 18))
#sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
#print(data.corr())
#plt.show()

# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
#
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
#data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
# #data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
# plt.legend(loc='upper right')     # legend = puts label into plot
# plt.xlabel('x axis')              # label = name of label
# plt.ylabel('y axis')
# plt.title('Line Plot')            # title = title of plot
# plt.show()

# data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')
# plt.xlabel('Attack')              # label = name of label
# plt.ylabel('Defence')
# plt.title('Attack Defense Scatter Plot')
# plt.show()

# data.Speed.plot(kind = 'hist',bins = 50,figsize = (12,12))
# plt.show()
# series = data['Defense']>200      # data['Defense'] = series
# print(data[series])

# print(data[np.logical_and(data['Defense']>200, data['Attack']>100 )])
# print(data.head() ) # head shows first 5 rows

# ----------------------------
##lineer regrassion
## residual= y()-y_head (predict tahmin edilmiş değer)
## MSE = sum(residual^2)/n n = sample sayısı yani 14
# ##Mse = mean squared error amaç min mse
# df = pd.read_csv('linear_regression_dataset.csv' , sep= ";")
# #
# # plt.scatter(df.deneyim, df.maas)
# # plt.xlabel("deneyim")
# # plt.ylabel("maas")
# # plt.show()
#
# #### lineer regreasssion simple
# ##lineer reg modal
# linear_reg = LinearRegression()
# ## values ile pandas objectten array'a dönüştürüdük. ##reshape ile 14,1 yapmış olduk
# x = df.deneyim.values.reshape(-1,1)
# y = df.maas.values.reshape(-1,1)
# ##ayarlamalar
# linear_reg.fit(x,y)
# ##prediction
# b0 = linear_reg.predict([[0]])
# print("b0:", b0)
#
#
# ## b0 modelimizi bulduk
# b0_ = linear_reg.intercept_
# print("b0_", b0_) ## y eksenini kestiği nokta
#
# ##linear coef ile b1 i bulabiliriz. yani eğimi
# ##eğimi veren yapı aşağıdadır.
# b1 = linear_reg.coef_
# print("b1",b1)
#
# # yani maas = 1663 + 1138* deneyim
#
# print(linear_reg.predict([[17]]))
# #visualizeline
# ## reshape yaparak scikitlearn reshape edilmiş veriyi görür.
# ##python 3.5 ve sonrasında predict ederken aşağıdaki gibi kullanmalısın bunu asla unutma
# array = np.array([[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]]).reshape(-1,1)
#
# ## tek tek predict edildi
# plt.scatter(x,y)
# y_head = linear_reg.predict(array)
# ##fit ettiğimiz linemizi çizdirdik
# plt.plot(array,y_head, color="red")
# plt.show()

# # ------------------------
# #multiple linear regreation
# #formul = LR = y = b0 + b1*x1 + b2*x2
# #maas = b0 + b1*deneyim + b2*yas
#
# df=pd.read_csv("multiple_linear_regression_dataset.csv" , sep =";")
# x = df.iloc[:,[0,2]] ## tüm satrıları al sıfır ve ikinci column'u al
# y = df.maas.values.reshape(-1,1)
# multiple_linear_regression = LinearRegression()
# multiple_linear_regression.fit(x,y)
# print("b0 ", multiple_linear_regression.intercept_)
# print("b1,b2", multiple_linear_regression.coef_)
#
# ##predict
# print(multiple_linear_regression.predict(np.array([[10,35],[5,35]])))
# # ------------------

# -----------------
# ##polynomal linear regration
# df = pd.read_csv("polynomial_regression.csv",sep = ";")
#
# y = df.araba_max_hiz.values.reshape(-1,1)
# x = df.araba_fiyat.values.reshape(-1,1)
#
# plt.scatter(x,y)
# plt.xlabel("araba_fiyat")
# plt.ylabel("araba_max_hiz")
#
#
# # linear regression =  y = b0 + b1*x
# # multiple linear regression   y = b0 + b1*x1 + b2*x2
#
# # linear regression
#
# lr = LinearRegression()
#
# lr.fit(x,y) ##newsquare errorun en az olacağı şekilde fit edilir.
#
# #predict
# y_head = lr.predict(x)
#
# plt.plot(x,y_head,color="red",label ="linear")
#
#
# print("10 milyon tl lik araba hizi tahmini: ",lr.predict([[10000]]))
#
#
#
# # polynomial regression =  y = b0 + b1*x +b2*x^2 + b3*x^3 + ... + bn*x^n
#
# polynomial_regression = PolynomialFeatures(degree = 2) ##polinomal olarak max derece yani üstel olarak en fazla x^2 ile çalışılır.
#
#
# x_polynomial = polynomial_regression.fit_transform(x) ## x^2 ye çevir modelimi oluştur.
#
#
# #  fit
# linear_regression2 = LinearRegression()
# linear_regression2.fit(x_polynomial,y)
#
# #
#
# y_head2 = linear_regression2.predict(x_polynomial)
#
# plt.plot(x,y_head2,color= "green",label = "poly")
# plt.legend()
# plt.show()

# # --------------------------
# ##Desicion Tree Regression
#
# df = pd.read_csv("decision+tree+regression+dataset.csv",sep = ";",header = None)
#
# x = df.iloc[:,0].values.reshape(-1,1)
# y = df.iloc[:,1].values.reshape(-1,1)
#
# # decision tree regression
#
# tree_reg = DecisionTreeRegressor()   # random sate = 0
# tree_reg.fit(x,y)
#
#
# tree_reg.predict([[5.5]])
# x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
# y_head = tree_reg.predict(x_)
# #  visualize
# plt.scatter(x,y,color="red")
# plt.plot(x_,y_head,color = "green")
# plt.xlabel("tribun level")
# plt.ylabel("ucret")
# plt.show()

###--------------------------------------