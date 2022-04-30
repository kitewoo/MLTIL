# Bag of Words (단어  가방 모형 : 단어 순서들은 전혀 고려하지 않고 오로지 빈도만을 수치화 표현하기)
# CountVectorizer : 피처추출 중 출현 빈도(frequency)로 텍스트를 벡터화 
from sklearn.feature_extraction.text import CountVectorizer
text = 'The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research.'


# stopword 정리 

# 1. nltk에 내장된 stopwords
import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords
sw = stopwords.words('english') #영어
print(len(sw))

# 2. Scikit-learn에서 제공하는 불용어 사용
cvect = CountVectorizer(stop_words='english')
print(cvect.fit_transform([text]).toarray()) #frequency로 변형하여 ndarray로 도출
print(cvect.vocabulary_)  # key : index 

# 3. 자체 제거
cvect = CountVectorizer(stop_words=['is', 'a', 'the']) #불용어 처리 가능.
'''아니면 직접 불용어 리스트 파일을 만들어놓고 불러와서 대입해도 된다.'''

#CountVectorizer의 파라미터들 

print(cvect.get_params()) 
'''분석대상 : 단어, dtype: numpy int64, ngram_range(1,1) 
n_gram은 예측에서 사용할 단어 (n-1)개를 설정한다. 예를 들어 3-gram은 뒤에서 2개의 단어를 가지고 예측에 활용한다.
An adorable little boy is spreading _ 이라는 예문에서 is spreading만 가지고 판단. 그리고 _에 올 단어의 비율을 train하여 확률적 선택을 하게됨.
문장 전체를 고려하는 모델보다는 정확도가 떨어질 수 밖에 없다. https://wikidocs.net/21692'''

#CountVectorizer로 벡터화 실행 
print(cvect.fit_transform([text]).toarray()) #frequency로 변형하여 ndarray로 도출
print(cvect.vocabulary_)  # key : index 



#TfidVectorizer(Term Frequency, Inverse Document Frequency)
'''Term Frequency 단일 문서 내 빈도 수
Inverse Document Frequency 전체 문서들 내에서 1회라도 출현했으면 1로 보고 이를 합산한 것의 역수. 즉, 문서군에서 중요도를 판단하기 위한 것'''

text = ['The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research.',
        'It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.']

from sklearn.feature_extraction.text import TfidfVectorizer
tvect = TfidfVectorizer(stop_words='english')
print(tvect.fit_transform(text).toarray())

