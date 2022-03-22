import numpy as np
import pandas as pd
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)

# pandas.read_csv()는 데이터 형식에 자료형이 포함되어 있지 않아 type 추론을 수행
# HDF5 나 Feather, msgpack 의 경우 데이터 형식에 자료형 포함
# type examples/ex1.csv
df = pd.read_csv('C:\pandas_dataset2\ex1.csv')
df

# 구분자를 쉼표로 지정
pd.read_table('C:\pandas_dataset2\ex1.csv', sep=',')
# type examples/ex2.csv
pd.read_csv('C:\pandas_dataset2\ex2.csv', header=None)
# 컬럼명 지정
pd.read_csv('C:\pandas_dataset2\ex2.csv', names=['a', 'b', 'c', 'd', 'message'])
# message 컬럼을 색인으로 하는 DAtaFrame 을 반환하려면 index_col 인자에 4 번째 컬럼
# 또는 'message'이름을 가진 컬럼을 지정하여 색인으로 만듦
names = ['a', 'b', 'c', 'd', 'message']
pd.read_csv('C:\pandas_dataset2\ex2.csv', names=names, index_col='message')
# type examples/csv_mindex.csv
# 계층적 색인 지정 시 컬럼 번호나 이름의 리스트를 넘긴다.
parsed = pd.read_csv('C:\pandas_dataset2\csv_mindex.csv', index_col=['key1', 'key2'])
parsed
# 구분자 없이 공백이나 다른 패턴으로 필드를 구분
list(open('C:\pandas_dataset2\ex3.txt'))
# 공백문자로 구분되어 있는 경우 정규표현식 \s+사용
result = pd.read_table('C:\pandas_dataset2\ex3.txt', sep='\s+')
result

# skiprows 를 이용하여 첫번째, 세번째, 네번째 로우를 건너뛴다.
pd.read_csv('C:\pandas_dataset2\ex4.csv', skiprows=[0, 2, 3])
# 텍스트파일에서 누락된 값은 표기되지 않거나(비어 있는 문자열) 구분하기 쉬운 특수한 문자로 표기
# 기본적으로 pandas 는 NA 나 NULL 처럼 흔히 통용되는 문자를 비어있는 값으로 사용
result = pd.read_csv('C:\pandas_dataset2\ex5.csv')
result
pd.isnull(result)
# na_values 옵션은 리스트나 문자열 집합을 받아서 누락된 값 처리
result = pd.read_csv('C:\pandas_dataset2\ex5.csv', na_values=['NULL'])
result
# 컬럼마다 다른 NA 문자를 사전값으로 넘겨서 처리
sentinels = {'message': ['foo', 'NA'], 'something': ['two']}
pd.read_csv('C:\pandas_dataset2\ex5.csv', na_values=sentinels)

# 큰 파일을 다루기 전에 pandas 의 출력 설정
pd.options.display.max_rows = 10
# 최대 10 개의 데이터 출력
result = pd.read_csv('C:\pandas_dataset2\ex6.csv')
result
# 처음 몇줄만 읽을 때 nrows 옵션 사용
pd.read_csv('C:\pandas_dataset2\ex6.csv', nrows=5)
# 파일을 여러 조각으로 나누어서 읽고 싶다면 chunksize 옵션으로 로우 개수 설정
chunker = pd.read_csv('C:\pandas_dataset2\ex6.csv', chunksize=1000)
# read_csv 에서 반환된 TetParser 객체를 이용해서 chunksize 에 따라 분리된 파일들을 순회할 수있다
# 예로 ex6.csv 파일을 순회하면서 'key'로우에 있는 값을 세어보려면 다음과 같이 한다.
chunker = pd.read_csv('C:\pandas_dataset2\ex6.csv', chunksize=1000)
tot = pd.Series([])
for piece in chunker:
 tot = tot.add(piece['key'].value_counts(), fill_value=0)
tot = tot.sort_values(ascending=False)
tot[:10]

# 데이터를 구분자로 구분한 형식으로 내보내기
data = pd.read_csv('C:\pandas_dataset2\ex5.csv')
data
data.to_csv('C:\pandas_dataset2\out1.csv')
# 다른 구분자 사용도 가능
import sys
data.to_csv(sys.stdout, sep='|')
# 결과에서 누락된 값은 비어 있는 문자열로 나타나는데 원하는 값으로 지정 가능
data.to_csv(sys.stdout, na_rep='NULL')
# 다른 옵션을 명시하지 않으면 로우와 컬럼 이름이 기록된다. 로우와 컬럼 이름을 포함하지 않을 경우 아래와 같이 사용
data.to_csv(sys.stdout, index=False, header=False)
# 컬럼의 일부분만 기록하거나 순서 지정
data.to_csv(sys.stdout, index=False, columns=['a', 'b', 'c'])
# Series 에도 to_csv 메서드 존재
dates = pd.date_range('1/1/2000', periods=7)
ts = pd.Series(np.arange(7), index=dates)
ts.to_csv('tseries.csv')
# type examples/tseries.csv

# type examples/ex7.csv
# pandas_read_table() 함수를 이용하여 대부분의 파일 형식을 불러 올 수 있다.
# csv 파일을 불러오는 경우
import csv
f = open('C:\pandas_dataset2\ex7.csv')
reader = csv.reader(f)
# 큰 따옴표가 제거된 튜플 얻을 수 있다.
for line in reader:
 print(line)
# 원하는 형태로 데이터를 넣을 수 있도록 하자.
# 파일을 읽어 줄 단위 리스트로 저장
with open('C:\pandas_dataset2\ex7.csv') as f:
 lines = list(csv.reader(f))
# 헤더와 데이터 구분
header, values = lines[0], lines[1:]
# 사전표기법과 로우를 컬럼으로 전치해주는 zip(*values)이용 데이터 컬럼 사전 만들기
data_dict = {h: v for h, v in zip(header, zip(*values))}
data_dict
# csv 파일은 다양한 형태로 존재할 수 있다. 다양한 구분자, 문자열을 둘러싸는 방법, 개행 문자 같은것들은
# csv.Dialect 를 상속받아 새로운 클래스를 정의해서 해결
class my_dialect(csv.Dialect):
 lineterminator = '\n'
 delimiter = ';'
 quotechar = '"'
 quoting = csv.QUOTE_MINIMAL
# reader = csv.reader(f, dialect=my_dialect)
reader = csv.reader('C:\pandas_dataset2\ex7.csv', dialect=my_dialect)
# 서브클래스를 정의하지 않고 csv.readr 에 키워드 인자로 각각의 csv 파일의 특징을 지정해서전달해도 된다.
# reader = csv.reader(f, delimiter='|')
reader = csv.reader('C:\pandas_dataset2\ex7.csv', delimiter='|')
# 사용가능한 옵션(csv.Dialect 의 속성)


# CSV 처럼 구분자로 구분된 파일을 기록하려면 csv.writer 를 이용하면 된다.
# csv.writer 는 이미 열린, 쓰기가 가능한 파일 개체를 받아서 csv.reader 와 동일한 옵션으로 파일을 기록
with open('mydata.csv', 'w') as f:
 writer = csv.writer(f, dialect=my_dialect)
 writer.writerow(('one', 'two', 'three'))
 writer.writerow(('1', '2', '3'))
 writer.writerow(('4', '5', '6'))
 writer.writerow(('7', '8', '9'))