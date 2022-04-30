
''' IMDB 사이트에 있는 영화평 데이터를 가지고 긍정,부정을 훈련시키고 테스트하고자 함. 모든 데이터의 저작권은 IMDB에 있음.'''
import numpy as np
import pandas as pd

df = pd.read_csv('c:/Users/tt/Desktop/machinelearning/07.텍스트분석/labeledTrainData.tsv', sep='\t', quoting=3) #seperate , quoting=3면 인용구 기호를 표시해준다. 
print(df.head())

# 텍스트 전처리 

df.review = df.review.str.replace('<br />', ' ') #띄어쓰기 태그는 공백으로 변환 
df.review = df.review.str.replace('[^A-Za-z]', ' ').str.strip() #알파벳을 제외한 나머지는 공백으로 변환
#print(df.review[0][:1000])

# Train/Test dataset 분리 

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split( 
    df.review, df.sentiment, stratify=df.sentiment, random_state=2022
)

# Countvectorize로 훈련,테스트 데이터를 str에서 vector로 변환

from sklearn.feature_extraction.text import CountVectorizer
cvect = CountVectorizer(stop_words='english')
cvect.fit(X_train) #학습훈련
X_train_cv = cvect.transform(X_train)
print(X_train_cv.shape)



