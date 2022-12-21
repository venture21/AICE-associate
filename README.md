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
