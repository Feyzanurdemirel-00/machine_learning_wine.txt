import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

Wine=pd.read_csv("Wine.csv") #read_csv fonksiyonu ile verisetini programa yükledik
w=Wine.copy()
w=Wine.dropna()
print(Wine.head()) #head satırı kullanarak verisetimizin ilk 5 satırınıı görüntüledik.
print(Wine.info()) #info ile Wine datasinin kısa bir özetini yazdırdık
print(Wine.shape) #shape ile verinin boyutunu yazdırdık (178,14)


sns.displot(Wine['Alcohol'])#buraya veri setimizde histogram grafiğini görmek istediğimiz sütun başlığını yazarız
plt.title("Alcohol-Frequency Histogram Grafiği")
plt.ylabel("Frequency")
plt.show()#istersek histogram grafiklerini alcohol,malic_asid....gibi böyle tek tek oluşturabiliriz
#bu kodu çalıştırdığımızda sadece Alcohol'ün histogram grafiği karşımıza çıkar.

sns.set()

Wine.hist(figsize=(10,20),color='blue')
plt.show() #istersek tüm veri setimizin histogram grafiklerini böyle yazdırabiliriz,
#programı çalıştırdığımızda tüm sütun başlıklarının histogram grafikleri karşımıza çıkar(14 tane)

#scatter plot matrixi bir değişkenin bir değişkenden veya değişkenlerden nasıl ve ne kadar etkilendiğini gösterir.
sns.set()
cols=['Alcohol','Malic_Acid']
sns.pairplot(Wine[cols])
plt.show()#bu kodu çalıştırdığımızda veri setimizdeki Alcohol değişkeninin  Malic Acid'den nasıl etkilendiğini görürüz

cols_tüm=['Alcohol','Malic_Acid','Ash','Ash_Alcanity','Magnesium']
sns.pairplot(Wine[cols_tüm])
plt.show()#istersek bu koddaki gibi birden fazla  değişkeninde scatter plot matrixini  inceleyebiliriz

y=w["Customer_Segment"]
X=w.drop(["Customer_Segment"] , axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)


#nb kullanırsak:
nb=GaussianNB()
nb_model=nb.fit(X_train,y_train)
print(nb_model.predict(X_test)[0:10]) #test vektörü üzerinde sınıflandırma gerçekleştirir
print(nb_model.predict_proba(X_test)[0:10])#test vektörü için olasılık tahminlerini döndürür.
y_pred=nb_model.predict(X_test)
print(confusion_matrix(y_test,y_pred)) #confusion matrixini yazdırdık
print(cross_val_score(nb_model,X_test,y_test,cv=10)) #çapraz doğrulama ile sonuçları yazdırdık
print(cross_val_score(nb_model,X_test,y_test,cv=10).mean())
print("nb accuracy score:")
print(accuracy_score(y_test,y_pred))#nb ile accuracy scoru 0.94 çıktı

#knn kullanırsak
knn=KNeighborsClassifier()
knn_model=knn.fit(X_train,y_train)
y_pred=knn_model.predict(X_test)
print(y_pred[0:10])
acc=accuracy_score(y_test,y_pred)
print(cross_val_score(knn_model,X_test,y_test,cv=10).mean())#çapraz doğrulama ile sonuçları yazdırdık
print("knn accuracy score:")
print(accuracy_score(y_test,y_pred))#knn ile accuracy score 0.7 çıktı


#decisiontreeclassifier kullanırsak:
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
print(cross_val_score(classifier,X_test,y_test,cv=10).mean())#çapraz doğrulama ile sonuçları yazdırdık
print("decision tree classifier accuracy score:")
print(accuracy_score(y_test,y_pred))# decision tree classifier ile accuracy score 0.87 çıktı

#logictic regression kullanırsak
loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X,y)
print(loj_model)
y_pred = loj_model.predict(X)
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))
loj_model.predict(X)[0:20] #test vektörü üzerinde sınıflandırma gerçekleştirir
loj_model.predict_proba(X)[0:10][:,0:2]#test vektörü için olasılık tahminlerini döndürür.
cross_val_score(loj_model, X_test, y_test, cv = 10).mean()#çapraz doğrulama ile sonuçları yazdırdık
print("Logistic regression accuracy score:")
print(accuracy_score(y, y_pred))#logistic regression ile score 0.97 çıktı..



