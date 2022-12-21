## 1. csv파일 불러오기
```
import pandas as pd
pd.read_csv('example.csv',sep=',')
```

## 2. 결측치 채우기
```
df.fillna('채울값', inplace=True)
```

## 3. 결측치가 있는 행 삭제하기
```
df.dropna(axis=0, inplace=True)
```

## 4. 특정 행 삭제하기
```
df.drop(삭제할 행 index, axis=0)
```

## 5. 특정 컬럼 삭제하기
```
df.drop(삭제할 컬럼명, axis=1)
# 삭제할 컬럼이 여러 개일 경우 리스트 사용
df.drop([컬럼명1,컬럼명2] axis=1)  
```
## 6. 데이터셋 분할하기

```
from sklearn.model_selection import train_test_split

# Feature데이터와 label데이터가 분리되어 있는 경우
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.2, random_state=42)

# Feature데이터와 label데이터가 하나의 데이터프레임에 같이 있는 경우
train_x, test_x, train_y, test_y = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.2, random_state=42)

# 데이터셋 분할 후, 분할된 데이터셋 확인
train_x.shape, test_x.shape, train_y.shape, test_y.shape
```
