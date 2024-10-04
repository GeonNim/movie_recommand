# 학습 모델

import pandas as pd
import numpy as np

from scipy.sparse import coo_matrix # COO 형식의 행렬을 만들기 위해 사용
from implicit.als import AlternatingLeastSquares # 알고리즘(머신러닝) 사용을 위해 import

import pickle # 모델 저장을 위해 import
import sys
import json
import threadpoolctl

threadpoolctl.threadpool_limits(1, "blas") # Python에서 실행되는 과학 계산 라이브러리들이 사용하는 다중 스레드(multithreading)를 제어하기 위한 함수 호출
# threadpool_limits(1, "blas")는 blas 라이브러리가 사용할 수 있는 스레드 수를 1로 제한한다는 의미입니다. 이렇게 하면 병렬 처리 대신 단일 스레드로 작업을 처리하게 됩니다.
# 병렬화된 연산이 오히려 성능을 저하시킬 수 있는 상황에서, 다중 스레드를 제한하여 더 나은 성능을 기대할 수 있습니다.
from scipy.sparse import coo_matrix, csr_matrix # coo_matrix : COO 형식의 행렬을 만들기 위해 사용, csr_matrix : CSR 형식의 행렬을 만들기 위해 사용

saved_model_fname = "model/finalized_model.sav"
data_fname = 'data/ratings.csv'
item_fname = 'data/movie_final.csv'
weight = 10 # 가중치 값 설정 (10으로 설정) : 사용자의 평점을 가중치로 사용하여 모델을 학습

def model_train():
    ratings_df = pd.read_csv(data_fname)
    ratings_df["userId"] = ratings_df["userId"].astype("category")
    ratings_df["movieId"] = ratings_df["movieId"].astype("category")

    # userid와 movieid를 category 형태로 변환
    # 어떤 유저가 어떤 영화에 얼마의 평점을 주었는지 행렬 형태로 표현해 주는 함수
    # 참조 : https://leebaro.tistory.com/entry/scipysparsecoomatrix
    # 참조에서 row에 해당하는 값이 movieId, col에 해당하는 값이 userId, data에 해당하는 값이 rating

    # Create a sparse matrix of all the item/user/counts triples
    rating_matrix = coo_matrix((ratings_df["rating"].astype(np.float32),
                                (ratings_df["movieId"].cat.codes.copy(),
                                ratings_df["userId"].cat.codes.copy(),),))
    
    # factors : 숫자가 클수록 기존 데이터에 대한 정확도는 높아지지만, 과적합의 위험이 있음. 이 경우 결과는 정확하지만, 새로운 데이터에 대한 예측력은 떨어질 수 있음.
    # 과적합 참조 : https://kimmaadata.tistory.com/31

    # regularization : 과적합을 방지하기 위한 정규화 항. 값이 클수록 정규화가 강해지고, 과적합이 줄어듦. 하지만, 값이 너무 크면 정확도가 떨어질 수 있음.

    # dtype : 데이터 타입. 기본값은 np.float64
    # float64로 설정하면 정확도가 높아지지만, 메모리 사용량이 늘어남.

    # iterations : 반복 횟수. 값이 클수록 정확도는 높아지지만, 시간이 오래 걸림.
 
    als_model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=50, dtype=np.float64)
    # Convert to CSR matrix
    rating_matrix_csr = rating_matrix.tocsr()

    # ALS 모델 학습
    als_model.fit(weight * rating_matrix_csr)

    pickle.dump(als_model, open(saved_model_fname, 'wb')) # wb : write binary
    return als_model

# item_id대신 
def calulate_item_based(item_id, items):
  loaded_model = pickle.load(open(saved_model_fname, 'rb')) # rb : read binary
  recs = loaded_model.similar_items(itemid=int(item_id), N=11)
  return [str(items[r]) for r in recs[0]]

def item_based_recommendation(item_id):
  ratings_df = pd.read_csv(data_fname)
  ratings_df["userId"] = ratings_df["userId"].astype("category")
  ratings_df["movieId"] = ratings_df["movieId"].astype("category")
  movies_df = pd.read_csv(item_fname)

  # cat.categories : 카테고리의 목록을 반환
  # astype("category")를 사용하여 특정 열을 카테고리형으로 변환할 수 있음
  # enumerate() 함수는 반복 가능한 객체(여기서는 movieId의 카테고리 리스트)에 인덱스를 추가 및 중복데이터 제외
  # dict() 함수는 키-값 쌍을 딕셔너리로 변환

  # 예시 데이터프레임
  # ratings_df = pd.DataFrame({
  #     'movieId': [100, 200, 100, 300, 200, 100]
  # })
  # ratings_df['movieId'] = ratings_df['movieId'].astype('category')

  # enumerate 사용 예시
  # print(dict(enumerate(ratings_df["movieId"].cat.categories)))

  # 출력 결과
  # {0: 100, 1: 200, 2: 300}

  items = dict(enumerate(ratings_df["movieId"].cat.categories))
  try:
    # cat.categories.get_loc() : 카테고리의 위치를 반환
    # item_id를 int로 변환하여 get_loc() 함수에 전달
    # get_loc() 함수는 해당 아이템의 위치를 반환
    parsed_id = ratings_df["movieId"].cat.categories.get_loc(int(item_id))
    result = calulate_item_based(parsed_id, items)
  except KeyError as e:
    result = []

  # result = ['1', '2', '3', 'item_id']이고 item_id = 2라면, 필터링된 결과는 [1, 3]
  # result 리스트 내의 값과 item_id의 타입이 다를 수 있으므로, 조건문에서 타입을 맞춘 후 비교
  # 즉, int(x) 가 아닌 x로 비교하면 item_id와 같은 아이템이 추천 결과에 포함될 수 있음
  result = [int(x) for x in result if int(x) != int(item_id)]

# isin(result)는 result 리스트에 포함된 영화 ID들만 선택하는 필터링 조건. 즉, result에 포함된 영화 ID가 movies_df의 movieId 열에 있는지 여부를 확인하고, 일치하는 행만 필터링.
  result_items = movies_df[movies_df["movieId"].isin(result)].to_dict("records")
  return result_items



# conda install -c conda-forge implicit

# 이 추천 알고리즘은 협업 필터링(Collaborative Filtering) 기반으로 동작하며, 특히 사용자 기반 협업 필터링(User-Based Collaborative Filtering) 방식을 사용.
# 주어진 설명에 따라, 1번 키(영화 ID)에 대해 4.5의 평점을 입력하면, 해당 사용자와 비슷한 평가 패턴을 가진 다른 사용자들의 정보를 기반으로 추천을 제공.


# 입력 평점 정보 처리:


# 사용자가 특정 아이템(여기서는 영화)에 대해 평점을 입력하면, 그 평점이 input_rating_dict로 전달.
# 예를 들어, input_rating_dict = {1: 4.5, 2: 3.0}는 사용자가 영화 ID 1에 대해 4.5점을, 영화 ID 2에 대해 3.0점을 줬다는 것을 의미.


# 사용자-아이템 행렬 구성:


# build_matrix_input() 함수는 입력된 평점을 기반으로 사용자-아이템 행렬을 리턴.
# 이 행렬은 희소 행렬로, 사용자가 평가한 아이템의 위치에 그 평점이 들어가고, 다른 위치는 0으로 채워짐.
# 이 과정에서 coo_matrix로 희소 행렬을 생성하여 사용자의 평가 정보를 벡터 형태로 변환.


# 유사 사용자 찾기:


# calculate_user_based() 함수는 사전에 학습된 추천 모델을 사용하여, 사용자가 입력한 평점 데이터를 기반으로 유사한 사용자들을 검색.
# 이 유사도는 사용자가 특정 영화에 대해 비슷한 평가를 한 다른 사용자들과의 유사성(코사인 유사도, 피어슨 상관계수 등)을 계산.
# user_items에는 현재 사용자의 평점 정보가 들어가고, 이 정보를 바탕으로 유사한 사용자들이 계산.


# 추천 계산:


# 유사한 사용자들을 기반으로 추천.
# 즉, 유사한 사용자들이 평점을 높게 준 영화들을 현재 사용자에게 추천.
# 이 과정에서 loaded_model.recommend() 함수가 사용되며, N개의 추천 영화를 반환.
# 이 함수는 일반적으로 협업 필터링 알고리즘을 기반으로 학습된 모델을 사용하여, 사용자가 아직 평가하지 않은 아이템들 중에서 유사한 사용자들이 선호한 아이템들을 추천.


# 결과 반환:


# 추천된 영화 목록은 아이템(영화) ID로 변환된 후, 다시 movie_df에서 해당 영화의 상세 정보를 조회하여 최종적으로 사용자에게 반환.


# 예시:
# 1번 키(영화 ID 1)에 대한 평점이 4.5인 경우 추천 방식:
# 사용자가 영화 ID 1에 대해 4.5점을 준 경우, 알고리즘은 이 영화를 선호하는 다른 사용자들과의 유사도를 계산.
# 이 영화와 비슷한 취향을 가진 다른 사용자들이 어떤 영화를 높게 평가했는지를 분석하여, 그 영화들을 추천.
# 만약 영화 ID 1을 좋아하는 다른 사용자들이 영화 ID 10과 영화 ID 15에 대해서도 높은 평점을 준 경우, 해당 영화들이 추천 목록에 포함될 가능성이 높음.



if __name__ == "__main__":
    command = sys.argv[1]

    if command == "item-based":
        item_id = sys.argv[2]
        # item-based 추천 시스템 실행
        print(json.dumps(item_based_recommendation(item_id)))
    else:
      print("Error: Invalid command error") 
      sys.exit(1)
  


    # model_train() # 데이터를 학습하여 모델을 생성하는 함수 호출. 처음 한번 실행 후 주석 처리. 이후 데이터가 추가됐을 때 주기적으로 실행