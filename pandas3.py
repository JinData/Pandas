import numpy as np
import pandas as pd
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)
# 결측치, 중복데이터 문자열 처리 , 분석적 데이터 변환에 대한 도구 설명

# pandas 객체의 모든 기술통계는 누락된 데이터를 배제하고 처리
# 산술 데이터에 한해 pandas 는 누락된 데이터를 실수값인 NaN 으로 취급
# 누락된 값을 쉽게 찾을 수 있다.
string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data

# 파이썬의 내장 None 값 또는 NA 값을 취급
string_data.isnull()
string_data[0] = None
string_data.isnull()

# Series 에 dropna 메서드를 적용하면 null 이 아닌 데이터와 색인값만 들어 있는 Series 반환
from numpy import nan as NA
data = pd.Series([1, NA, 3.5, NA, 7])
data.dropna()

# 위의 코드와 동일한 코드
data[data.notnull()]

# NA 값을 하나라도 포함하고 있는 로우나 컬럼을 제외시킬 수 있다.
data = pd.DataFrame([[1., 6.5, 3.], [1., NA, NA],
 [NA, NA, NA], [NA, 6.5, 3.]])
cleaned = data.dropna()
data
cleaned

# how='all'옵션을 이용하면 모두 NA 값인 로우만 제외
data.dropna(how='all')

# axis = 1 옵션을 이용하여 모두 NA 값인 컬럼을 제외시키는 방법
data[4] = NA
data
data.dropna(axis=1, how='all')

# 몇 개 이상의 값이 들어 있는 로우만 살펴보고 싶다면 thresh 인자에 원하는 값 설정
df = pd.DataFrame(np.random.randn(7, 3))
df.iloc[:4, 1] = NA
df.iloc[:2, 2] = NA
df
df.dropna()
df.dropna(thresh=2) # 2 개 이상의 값이 있는 로우 출력

# 누락된 값을 제외시키지 않고 다른 값으로 대체할 때 fillna 메서드 사용
df.fillna(0)
df.fillna({1: 0.5, 2: 0}) # 컬럼 1 에는 0.5, 컬럼 2 에는 0 대체
# fillna 는 새로운 객체를 반환하지만 기존 객체를 변경할 수도 있다.
_ = df.fillna(0, inplace=True)
df
# 재색인에서 사용한 보간메서드는 fillna 메서드에서도 사용가능하다.
df = pd.DataFrame(np.random.randn(6, 3))
df.iloc[2:, 1] = NA
df.iloc[4:, 2] = NA
df
df.fillna(method='ffill')
df.fillna(method='ffill', limit=2)
# Series 의 평균값이나 중간값을 전달할 수도 있다.
data = pd.Series([1., NA, 3.5, NA, 7])
data.fillna(data.mean())

# 중복된 로우 발견 예제
data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'],
 'k2': [1, 1, 2, 3, 3, 4, 4]})
data
# DataFrame 의 duplicated 메서드는 각 로우가 중복인지 아닌지 알려주는 불리언 Series 를 반환
data.duplicated()
# drop_duplicates()는 duplicated 배열이 false 인 DataFrame 을 반환
data.drop_duplicates()
# 새로운 컬럼을 하나 추가하고 'k1'컬럼을 기반해서 중복을 걸려내려는 경우
data['v1'] = range(7)
data
data.drop_duplicates(['k1'])
# duplicated 와 drop_duplicates 는 기본적으로 처음 발견된 값을 유지
# keep = 'last'옵션을 넘기면 마지막으로 발견된 값을 반환
data.drop_duplicates(['k1', 'k2'], keep='last')

# DataFrame 의 컬럼이나 Series, 배열 내의 값을 기반으로 데이터의 형태를 변환
data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon',
 'Pastrami', 'corned beef', 'Bacon',
 'pastrami', 'honey ham', 'nova lox'],
 'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data
# 해당 육류가 어떤 동물의 고기인지 알려줄 수 있는 컬럼을 하나 추가
# 육류별 동물을 담고 있는 사전 데이터 작성
meat_to_animal = {
 'bacon': 'pig',
 'pulled pork': 'pig',
 'pastrami': 'cow',
 'corned beef': 'cow',
 'honey ham': 'pig',
 'nova lox': 'salmon'
}
data
# 육류 이름에 대소문자가 섞여 있는 문제를 해결
# str.lower 메서드를 사용해서 모두 소문자로 변경
lowercased = data['food'].str.lower()
lowercased
data['animal'] = lowercased.map(meat_to_animal)
data
# map 메서드를 이용하여 데이터의 요소별 변환 가능
data['food'].map(lambda x: meat_to_animal[x.lower()])

# fillna 메서드를 사용해서 누락된 값을 채우는 것은 값 치환 작업이라 볼 수 있음
# map 메서드는 한 객체 안에서 값의 부분집합을 변경하는데 사용
# replace 메서드는 같은 작업을 간단하고 유연한 방법 제공
data = pd.Series([1., -999., 2., -999., -1000., 3.])
data
# -999 는 누락된 데이터를 나타내기 위한 값
# replace 메서드를 이용하여 NA 값으로 치환한 새로운 Series 생성
data.replace(-999, np.nan)
# 여러 개의 값을 한 번에 치환하려면 하나의 값 대신 치환하려는 값의 리스트 사용
data.replace([-999, -1000], np.nan)
# 치환하려는 값마다 다른 값으로 치환하려면 누락된 값 대신 새로 지정할 값의 리스트 사용
data.replace([-999, -1000], [np.nan, 0])
# 두개의 리스트 대신 dict 를 이용하는 것도 가능
data.replace({-999: np.nan, -1000: 0})

# 축 이름 도 함수나 새롭게 바꿀 값을 이용하여 변환 가능
# 예제
data = pd.DataFrame(np.arange(12).reshape((3, 4)),
 index=['Ohio', 'Colorado', 'New York'],
 columns=['one', 'two', 'three', 'four'])
data
# 축 색인에도 map 메서드 사용
transform = lambda x: x[:4].upper()
data.index.map(transform)
# 대문자로 변경된 축 이름을 DataFrame 의 index 에 바로 대입
data.index = data.index.map(transform)
data
# 원래 객체를 변경하지 않고 새로운 객체 생성 시 rename 메서드 사용
data.rename(index=str.title, columns=str.upper)
# dic 객체를 이용하여 축 이름 중 일부만 변경 가능
data.rename(index={'OHIO': 'INDIANA'},
 columns={'three': 'peekaboo'})
data # 원본 유지
# rename 메서드를 사용하면 DataFrame 을 직접 복사해서 index 와 columns 속성을 갱신할 필요 없이바로 변경 가능
# 원본 데이터를 바로 변경하려면 inplace=True 옵션 사용
data.rename(index={'OHIO': 'INDIANA'}, inplace=True)
data # 원본 수정

# 연속형 데이터를 개별로 분할하거나 그룹별로 나누기도 한다.
# 수업에 참여하는 학생 그룹 데이터가 있고 나이대에 따라 분류한다고 가정
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
# 이 데이터를 pandas 의 cut() 함수를 이용하여 18-25, 26-35, 35-60, 60 이상 그룹으로나누어보자
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
cats

# pandas 에서 반환하는 객체는 Categorical 이라는 특수한 객체
# Catergorical 객체는 codes 속성에 있는 ages 데이터에 대한 카테고리 이름을
# catergires 라는 배열에 내부적으로 담고 있다.
cats.codes
cats.categories
pd.value_counts(cats)

# right=False 로 설정하여 중괄호 대신 대괄호 쪽이 포함되지 않도록 변경 가능
pd.cut(ages, [18, 26, 36, 61, 100], right=False)
# labels 옵션 사용으로 그룹의 이름 추가 가능
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=group_names)
# cut 함수에 명시적으로 그룹의 경계값을 넘기지 않고 그룹의 개수를 넘겨주면
# 데이터에서 최소값과 최대값을 기준으로 균등한 길이의 그룹을 자동으로 계산

 #cut 함수에 명시적으로 그룹의 경계값을 넘기지 않고 그룹의 개수를 넘겨주면
# 데이터에서 최소값과 최대값을 기준으로 균등한 길이의 그룹을 자동으로 계산
# 4 개의 그룹으로 나누는 경우
# precision = 2 옵션은 소수점 아래 2 자리까지 표시
data = np.random.rand(20)
pd.cut(data, 4, precision=2)
# cut()함수를 이용하면 데이터의 분산에 따라 각각의 그룹마다 데이터 수가 다른게 나뉘는 경우가많은데
# 같은 크기의 그룹으로 나눌때는 표준 변위치를 사용하는 qcut() 함수 이용
data = np.random.randn(1000) # Normally distributed
cats = pd.qcut(data, 4) # Cut into quartiles
cats
# 변위치를 직접 지정 가능(변위치는 0~1)
pd.value_counts(cats)
pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])

# 이상값을 제외하거나 다른 값으로 대체
# 예제
data = pd.DataFrame(np.random.randn(1000, 4))
data.describe()
# DataFrame 의 한 컬럼에서 절대값이 3 을 초과하는 값을 찾기
col = data[2]
col[np.abs(col) > 3]
# 절대값이 3 을 초과하는 값이 들어있는 모든 로우를 선택하려면 불리언 DataFrame 에서 any 메서드사용
data[(np.abs(data) > 3).any(1)]
# -3 이나 3 을 초과하는 값을 -3 또는 3 으로 지정 가능
data[np.abs(data) > 3] = np.sign(data) * 3
data.describe()
# np.sign(data)는 data 값이 양수인지 음수인지에 따라 1 이나 -1 이 담긴 배열 반환
np.sign(data).head()

# -3 이나 3 을 초과하는 값을 -3 또는 3 으로 지정 가능
data[np.abs(data) > 3] = np.sign(data) * 3
data.describe()
# np.sign(data)는 data 값이 양수인지 음수인지에 따라 1 이나 -1 이 담긴 배열 반환
np.sign(data).head()

# numpy.random.permutation()함수를 이용하면 Series 나 DataFrame 의 로우를 임의로 재배치가능
# 순서를 바꾸고 싶은 만큼의 길이를 permutation()함수로 넘기면 바뀐 순서가 담긴 정수 배열 생성
df = pd.DataFrame(np.arange(5 * 4).reshape((5, 4)))
df
sampler = np.random.permutation(5)
sampler
# 이 배열은 iloc 기반의 색인이나 take()함수에서 사용 가능
df
df.take(sampler) # sampler 의 index 순서
# 치환없이 일부만 임의로 선택하려면 Series 나 DataFrame 의 sample 메서드 사용
df.sample(n=3)
# 복원추출을 허용하며 표본을 치환을 통해 생성하려면 replace=True 옵션 사용
choices = pd.Series([5, 7, -1, 6, 4])
draws = choices.sample(n=10, replace=True)
draws

# 분류값을 '더미'나 '표시자' 행렬로 전환
# DataFrame 의 한 컬럼에 k 가지 값이 있을때 k 개의 컬럼이 있는 DataFrame 이나 행렬을 만들고
# 값으로 1 과 0 으로 채우는 것
# pandas 의 get_dummies()가 이런 역할을 수행함.
df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
 'data1': range(6)})
pd.get_dummies(df['key'])

# 표시자 DataFrame 안에 있는 컬럼에 접두어(prefix)를 추가한 후 다른 데이터와 병합하고 싶을 때
# get_dummies()함수의 prefix 인자를 사용
dummies = pd.get_dummies(df['key'], prefix='key')
df_with_dummy = df[['data1']].join(dummies)
df_with_dummy

# DataFrame 의 한 로우가 여러 카테고리에 속한 경우
# 예제
mnames = ['movie_id', 'title', 'genres']
# movies = pd.read_table('datasets/movielens/movies.dat', sep='::',
# header=None, names=mnames)
movies = pd.read_table('C:\pandas_dataset2\movies.dat', sep='::',header=None, names=mnames)
movies[:10]

# 각 장르마다 표시자 값을 추가하기 위해서 데이터 묶음에서 유일한 장르 목록 추출
all_genres = []
for x in movies.genres:
 all_genres.extend(x.split('|'))
genres = pd.unique(all_genres)
genres

# 표시자 DataFrame 생성을 위하여 0 으로 초기화된 DataFrame 생성
zero_matrix = np.zeros((len(movies), len(genres)))
dummies = pd.DataFrame(zero_matrix, columns=genres)

# 각 영화를 순회하면서 dummies 의 가가 로우의 항목을 1 로 설정
# 각 장르의 컬럼 색인을 계산하기 위해 dummies.columns 사용
gen = movies.genres[0]
gen.split('|')
dummies.columns.get_indexer(gen.split('|'))

# iloc 를 이용하여 색인에 맞게 값을 대입
for i, gen in enumerate(movies.genres):
 indices = dummies.columns.get_indexer(gen.split('|'))
dummies.iloc[i, indices] = 1

# movies 와 조합
movies_windic = movies.join(dummies.add_prefix('Genre_'))
movies_windic.iloc[0]
# get_dummies 와 cut 같은 이산함수를 잘 조합하면 통계 application 에서 유용하게 사용 가능
np.random.seed(12345)
values = np.random.rand(10)
values
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
pd.get_dummies(pd.cut(values, bins))

# 쉼표로 구분된 문자열은 split 메서드를 이용하여 분리
val = 'a,b, guido'
val.split(',')
# split 메서드는 공백문자(줄바꿈 문자 포함)를 제거하는 strip 메서드와 조합하여 사용 가능
pieces = [x.strip() for x in val.split(',')]
pieces
# 분리된 문자열은 더하기 연산을 사용하여 ::문자열과 합칠 수도 있다.
first, second, third = pieces
first + '::' + second + '::' + third
# 이 방법은 실용적이거나 범용적이지 않음.
# 보다 나은 방법은 리스트나 튜플을 ::문자열의 join 메서드로 전달하는 것
'::'.join(pieces)
# 일치하는 부분문자열의 위치를 찾는 방법
# in 예약어를 사용하면 일치하는 부분문자열을 쉽게 찾을 수 있다.
'guido' in val
val.index(',')
val.find(':')
# find 와 index 의 차이점:
# index 의 경우 문자열을 찾지 못하면 예외 발생
# find 의 경우 -1 을 반환
val.index(':')
# count 메서드는 특정 부분문자열이 몇 건 발견되었는지 반환
val.count(',')
# replace 메서드는 찾아낸 패턴을 다른 문자열로 치환
# 대체할 문자열로 비어있는 문자열을 설정하여 패턴을 삭제하기 위한 방법으로 자주 사용
val.replace(',', '::')
val.replace(',', '')

# 여러가지 공백 문자(탭, 스페이스)가 포함된 문자열을 나누고 싶다면
# 하나 이상의 공백 문자를 의미하는 \s+를 사용하여 문자열 분리
import re
text = "foo bar\t baz \tqux"
re.split('\s+', text)
# re.split('\s+', text)를 사용하면 정규 표현식이 컴파일되고 split 메서드가 실행
# re.compile 로 직접 정규 표현식을 컴파일하고 얻은 정규 표현식 객체를 재사용하는 것도 가능
regex = re.compile('\s+')
regex.split(text)
# 정규 표현식에 매칭되는 모든 패턴의 목록이 필요한 경우 findall 메서드 사용
regex.findall(text)
# 정규표현식에서 \문자가 이스케이프되는 문제를 피하려면 raw 문자열 표기법 사용
# 'C:\\x' 대신 r'C:\x' 사용
# 같은 정규 표현식을 다른 문자열에도 적용해야 한다면 re.comile 을 이용하여 정규 표현식 색체를만들어 쓰는 방법 추천
# CPU 사용량 절약 가능

# 이메일 주소를 검사하는 정규 표현식
text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
# re.IGNORECASE makes the regex case-insensitive
regex = re.compile(pattern, flags=re.IGNORECASE)
# finall 메서드 사용 이메일 주소의 리스트 생성
regex.findall(text)

# search 는 텍스트에서 첫 번째 이메일 주소만 찾아준다.
# match 는 그 정규 표현 패턴이 문자열 내에서 위치하는 시작점과 끝점만을 알려준다.
m = regex.search(text)
m
text[m.start():m.end()]

# regex.match 는 None 반환. 왜냐하면 그 정규 표현 패턴이 문자열의 시작점에서부터 일치하는지검사하기 때문
print(regex.match(text))
# sub 메서드는 찾은 패턴을 주어진 문자열로 치환하여 새로운 문자열 반환
print(regex.sub('REDACTED', text))
# 이메일 주소를 찾아서 사용자 이름, 도메인 이름, 도메인 접미사 로 나눠야 한다면 객 패턴을 괄호로묶어준다.
pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(pattern, flags=re.IGNORECASE)
# match 객체를 이용하면 groups 메서드로 각 패턴 컴포넌트의 튜플을 얻을 수 있다.
m = regex.match('wesm@bright.net')
m.groups()
# 패턴에 그룹이 존재한다면 findall 메서드는 튜플의 목록 반환
regex.findall(text)

# sub 역시 마찬가지로 \1, \2 같은 특수한 기호를 사용하여 각 패턴 그룹에 접근할 수 있다.
# \1 : 첫 번째로 찾은 그룹을 의미; \2 : 두번째로 찾은 그룹 의미
print(regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text))

# 문자열을 담고 있는 컬럼에 누락된 값이 포함된 경우
data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com',
 'Rob': 'rob@gmail.com', 'Wes': np.nan}
data = pd.Series(data)
data
data.isnull()
# 문자열과 정규 표현식 메서드는 data.map 을 사용하여 각 값에 적용(lambda 혹은 다른 함수 이용) 할 수 있지만
# NA 값을 만나면 실패함
# 이런 문제를 해결하기 위해 Series 에는 NA 값을 건너뛰도록 하는 문자열 처리 메서드str.contains 가 있음.
data.str.contains('gmail')
# 정규표현식을 IGNORECASE 같은 re 옵션을 함께 사용하는 것도 가능
# re.IGNORECASE : 대/소문자를 구분하지 않는 일치를 수행 (예, x = X)
pattern
data.str.findall(pattern, flags=re.IGNORECASE)
# 벡터화된 요소를 꺼내오는 방법: str.get 이용 또는 str 속성의 색인 이용
matches = data.str.match(pattern, flags=re.IGNORECASE)
matches
# 내재된 리스트의 원소에 접근하기 위해 색인 이용
matches.str.get(1) #??
matches.str[0] #??
data.str.get(1)
data.str[0]
# 문자열을 잘라내기
data.str[:5]



























































