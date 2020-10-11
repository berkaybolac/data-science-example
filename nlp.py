import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


data = pd.read_csv(r"gender_classifier.csv", encoding = "latin1")
data = pd.concat([data.gender, data.description], axis=1)
data.dropna(axis=0, inplace=True) ## inplace Data = data.dropna demekle aynı şey
##axis tek satır bazlı çalışmayı sağlar
## dropna NaN verileri drop eder
data.gender = [1 if each == "female" else 0 for each in data.gender]

first_description = data.description[4]
description = re.sub("[^a-zA-Z]"," ",first_description)
description = description.lower()

##description = description.split() ## shouldnt gibi keliemeleri ikiye ayıramaz split yerine tokenizer kullanılabilir.
description = nltk.word_tokenize(description)
description = [word for word in description if not word in set(stopwords.words("english"))]
lemma = nltk.WordNetLemmatizer()
description = [lemma.lemmatize(word) for word in description]
description = " ".join(description)
description_list = []
for description in data.description:
    description = re.sub("[^a-zA-Z]", " ", description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    #description = [word for word in description if not word in set(stopwords.words("english"))]
    lemma = nltk.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)

max_features = 5000 ## en çok kullandığım 500'ü seç
count_vectorizer = CountVectorizer(max_features= max_features, stop_words="english")
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()
print("ensik kullanılan {} kelimeler {}".format(max_features, count_vectorizer.get_feature_names()))

y = data.iloc[:,0].values
x = sparce_matrix
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state=42)

nb = GaussianNB()
nb.fit(x_train,y_train)

y_pred = nb.predict(x_test)
print("accuracy: ", nb.score(y_pred.reshape(-1,1),y_test))