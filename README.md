# 0. Python
## 0-1. 패키지 설치하기
```
pip install [패키지명]
```

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
pd.get_dummies(data=df, columns=['컬럼명'])
```

# 2. Scikit-learn

## 2-1. 데이터셋 분할하기

```
from sklearn.model_selection import train_test_split

# Feature데이터와 label데이터가 분리되어 있는 경우
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.2, random_state=42)

# Feature데이터와 label데이터가 하나의 데이터프레임에 같이 있는 경우
train_x, test_x, train_y, test_y = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.2, random_state=42)

# Feature데이터와 label데이터가 하나의 데이터프레임에 같이 있는 경우, (레이블까지 고려하여) 데이터셋 균등하게 나누기
train_x, test_x, train_y, test_y = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.2, stratify=df['label'], random_state=42)

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

## 2-4. 머신러닝 모델 - 분류 : 랜덤포레스트 분류기
```
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=123)
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
```

## 2-5. 머신러닝 모델 - 분류 : 로지스틱 회귀
```
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
```

## 2-6. 머신러닝 모델 - 분류 : K-Nearest Neighbor
```
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
```

## 2-7. 머신러닝 모델 - 분류 : DecisionTree
```
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(randon_state=42)
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
```

## 2-8. 머신러닝 모델 - 분류 : xgboost
```
pip install xgboost
```

```
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=3, random_state=42) 
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
```

## 2-9. 머신러닝 모델 - 분류 : LGBM
```
pip install lightgbm
```

```
from lightgbm import LGBMClassifier

model = LGBMClassifier(n_estimators=3, random_state=42) 
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

## 4-4. 딥러닝 모델 - 콜백 함수 설정
```
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# val_loss 모니터링해서 성능이 5epoch동안 val_loss가 낮아지지 않으면 조기 종료
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5)

# val_loss 가장 낮은 값을 가질때마다 모델저장
check_point = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

# fit()함수의 callbacks에 전달할 리스트
callbacks = [early_stop, check_point]
```

## 4-5 딥러넝 모델 - 학습
```
history = model.fit(x=X_train, y=y_train, epochs=50 , batch_size=20,
          validation_data=(X_test, y_test), verbose=1, callbacks=callbacks)
```

## 4-6. 딥러닝 모델 - 성능 평가 - loss 그래프 그리기

```
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend(['acc', 'val_acc'])
plt.show()
```







