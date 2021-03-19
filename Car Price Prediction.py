#!/usr/bin/env python
# coding: utf-8

# # Araba Fiyatlarının Tahmin Edilmesi

# Kullanılacak kütüphanelerin import edilmesi

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn


# Excel dosyasının okutulması

# In[3]:


dataFrame=pd.read_excel("merc.xlsx")


# Datasetin yapısının incelenmesi(İlk 5 eleman ile)

# In[4]:


dataFrame.head()


# Datasetin istatiksel analizinin yapılması

# In[5]:


dataFrame.describe()


# Datasette boş olan verilerin incelenmesi

# In[6]:


dataFrame.isnull().sum()


# Fiyatların hangi dağılıma benzediği ve outlier değerlerin incelenmesi(grafik yüksekliğinin ve görünümünün ayarlanması)

# In[29]:


sbn.displot(dataFrame["price"], height=8, aspect=1)


# Year değişkenine bakarak yıllara göre fiyat analizi yapılması(X ekseninde bulunan verilerin 45 derece döndürülmesi)

# In[30]:


sbn.countplot(dataFrame["year"])
plt.xticks(rotation=-45)


# Değişkenler arasındaki ilişkilere bakılması

# In[31]:


dataFrame.corr()


# Price değişkeninin diğer verilerle korelasyon ilişkisine bakılması

# In[33]:


dataFrame.corr()["price"].sort_values()


# Serpilme diyagramıyla verilerin nerede toplanıldığına bakılması

# In[34]:


sbn.scatterplot(x="mileage", y="price", data=dataFrame)


# Yüksek fiyatlı 20 ürünün özelliklerine göre sıralanması

# In[35]:


dataFrame.sort_values("price",ascending=False).head(20)


# Güven aralığını %99 alarak 1970 yılından başlayarak 131 verinin silinmesi(13000 verinin %1i silinerek anlamlı veri seti elde edilmeye çalışılmıştır.)

# In[36]:


doksanDokuzDF=dataFrame.sort_values("price",ascending=False).iloc[131:]


# 12869 verinin istatistiksel özelliklerine bakılması

# In[37]:


doksanDokuzDF.describe()


# 12869 verideki fiyatların dağılımına bakılması

# In[40]:


sbn.displot(doksanDokuzDF["price"])


# Dataların yıl değişkenine göre fiyatlandırmasına bakılarak gruplandırılması

# In[41]:


dataFrame.groupby("year").mean()["price"]


# Yukarıdaki gruplamada 1970 yılındaki verinin diğer verilere göre tutarsız olduğu görülüyor. Bundan dolayı veri çıkartılırsa daha anlamlı bir veri setiyle işlem yapılabilir. 

# In[42]:


dataFrame[dataFrame.year!=1970].groupby("year").mean()["price"]


# DataFramein %99 güven aralığıyla çalışması

# In[43]:


dataFrame=doksanDokuzDF


# In[44]:


dataFrame.describe()


# 1970 datasını çıkartma işleminin gerçekleşmesi

# In[45]:


dataFrame=dataFrame[dataFrame.year!=1970]


# In[46]:


dataFrame.groupby("year").mean()["price"]


# In[24]:


dataFrame.head()


# Transmission ifadesi string olduğu için çıkartılmasının daha uygundur.

# In[48]:


dataFrame=dataFrame.drop("transmission", axis=1)


# In[49]:


dataFrame.head()


# Bağımlı ve bağımsız değişkenlerin belirlenmesi

# In[50]:


y=dataFrame["price"].values
x=dataFrame.drop("price",axis=1).values


# Test ve eğitim değişkenlerinin oluşturulması ve import edilmesi

# In[51]:


from sklearn.model_selection import train_test_split


# Bir tahmin fonksiyonunun parametrelerini öğrenmek ve onu aynı veriler üzerinde test etmek overfittinge yol açacağı için verilerin %30uyla test verisi oluşturarak model eğitilmelidir.

# In[52]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=10)


# In[53]:


len(x_train)


# In[54]:


len(y_train)


# Ölçeklendirme işleminin yapılması(Bulunan verilerin -1 ile 1 arasına çekilmesi)

# In[56]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[57]:


x_train=scaler.fit_transform(x_train)


# In[58]:


x_test=scaler.transform(x_test)


# Model ve katman kütüphanelerinin import edilmesi

# In[60]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
x_train.shape


# 1 giriş katmanı,5 ara katman, 1 tane de çıkış katmanı oluşturulması. Modelin adam optimizeri ve kayıbın  hata kareler toplamıyla oluşturulması.

# In[61]:


model=Sequential()
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")


# Modelin eğitilmesi için batch size ve epochs(Devir sayısı) belirlenmesi

# In[62]:


model.fit(x=x_train, y=y_train, validation_data=(x_test,y_test), batch_size=250,epochs=300)


# In[63]:


kayip=pd.DataFrame(model.history.history)


# In[64]:


kayip.head()


# Loss ve Validation Loss birlikte azalarak hareket ediyor mu? Bunun için grafik çizilebilir.

# In[65]:


kayip.plot()


# In[66]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[67]:


tahminDizisi=model.predict(x_test)


# Tahmin dizisi ile y test arasındaki absolute farka bakılır.

# In[69]:


mean_absolute_error(y_test,tahminDizisi)


# Dataframe özelliğinden priceın ortalama değerine bakılır. Mean absolute değeri(3153 para birimi) ortalama değerden(24074 para birimi) ne kadar sapabilir.

# In[44]:


dataFrame.describe()


# Aşağıdaki grafikte linear regression görülmekte fakat küçük sapmalar mevcut eğer dilerseniz bu sapmalar için gerekli düzenlemeler yapabilirsiniz.(Epoch sayısı arttırma gibi)

# In[70]:


plt.scatter(y_test,tahminDizisi)
plt.plot(y_test,y_test,"g-*")


# 3. Elemanın fiyatına bakalım.

# In[71]:


dataFrame.iloc[2]


# 3. elemanı çıkartarak yeni bir veri serisi oluşturalım.

# In[73]:


yeniArabaSeries=dataFrame.drop("price", axis=1).iloc[2]


# Yeni veri serisini ölçeklendirelim.

# In[74]:


yeniArabaSeries=scaler.transform(yeniArabaSeries.values.reshape(-1,5))


# Tahmin edilen fiyat 63477 para birimi oluyor. Yeni veri serisi oluştururken çıkarttığımız eleman ise 65980 para birimiydi. Tahmin işlemi gerçekleştirilmiş oldu.

# In[77]:


model.predict(yeniArabaSeries)


# In[ ]:




