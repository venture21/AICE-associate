# 1. Pandas 

## 1-1. csv파일 불러오기
```
import pandas as pd
pd.read_csv('example.csv',sep=',')
```

## 1-2. 결측치 채우기
```
df.fillna('채울값', inplace=True)
```

## 1-3. 결측치가 있는 행 삭제하기
```
df.dropna(axis=0, inplace=True)
```

## 1-4. 특정 행 삭제하기
```
df.drop(삭제할 행 index, axis=0)
```

## 1-5. 특정 컬럼 삭제하기
```
df.drop(삭제할 컬럼명, axis=1)
# 삭제할 컬럼이 여러 개일 경우 리스트 사용
df.drop([컬럼명1,컬럼명2] axis=1)  
```
## 1-6 더미 변수 타입으로 변환(One-hot encoding)
```
pd.get_dummies(data=df, columns=['MultipleLines'])
```

# 2. Scikit-learn

## 2-1. 데이터셋 분할하기

```
from sklearn.model_selection import train_test_split

# Feature데이터와 label데이터가 분리되어 있는 경우
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.2, random_state=42)

# Feature데이터와 label데이터가 하나의 데이터프레임에 같이 있는 경우
train_x, test_x, train_y, test_y = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.2, random_state=42)

# 데이터셋 분할 후, 분할된 데이터셋 확인
train_x.shape, test_x.shape, train_y.shape, test_y.shape
```

## 2-2. 데이터 정규화 - MinMaxScaler
```
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)
```
## 2-3. 데이터 정규화 - StandardScaler
```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)
```

## 2-4. 머신러닝 모델 - 랜덤포레스트 분류기
```
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=123)
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
```
## 2-5. 머신러닝 모델 - 로지스틱 회귀
```
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
```

# 4. Tensorflow(Keras)

## 4-1. 딥러닝 모델 - 심층신경망 만들기(회귀 모델)
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.random import set_seed

set_seed(100)

col_num = train_x.shape[1]

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(col_num,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['mae'])      
model.summary()
```
## 4-2. 딥러닝 모델 - 심층신경망 만들기(이진 분류 모델)
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.random import set_seed

set_seed(100)

col_num = train_x.shape[1]

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(col_num,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])      
model.summary()
```
## 4-3. 딥러닝 모델 - 심층신경망 만들기(다중 분류 모델)
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.random import set_seed

set_seed(100)
class_num = 2

# 회귀 모델의 경우
col_num = train_x.shape[1]

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(col_num,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(class_num, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])      
model.summary() 
```









