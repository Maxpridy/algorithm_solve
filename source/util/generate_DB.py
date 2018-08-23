import csv
import itertools
import random
import numpy as np
import os
import json
from source.util import get_label

def generate_DB(newDataPath=None, rawDataPath=None, sample_for_line=50, hole_value=-1):
    if rawDataPath is None:
        rawDataPath = '../../data/16col_original_25000line_trainset.csv'
    if newDataPath is None:
        newDataPath = '../../data/samsung_db'

    l2n = get_label.get_labels()
    labels = ['주야', '요일', '발생지시도', '발생지시군구', '사고유형_대분류', '사고유형_중분류', '법규위반', '도로형태_대분류', '도로형태', '당사자종별_1당_대분류',
              '당사자종별_2당_대분류']
    l2n_idx = [l2n[label] for label in labels]
    with open(rawDataPath, newline='') as f:
        data = csv.reader(f, delimiter=',', quotechar='|')
        print('raw data loaded')

        raw = []
        hole = []
        for line_idx, line in enumerate(data):
            if line_idx % 1000 == 0:
                print('processing... {}/{}'.format(line_idx+1, 25001))
            candidates = itertools.combinations(range(len(line)), 3) # 3개 조합 계산
            seleted = random.sample(list(candidates), sample_for_line) # 모든 경우의 수 중 X개만 선택
            for i in [2, 3, 4, 5, 6]:
                line[i] = int(line[i])
            for idx, element in enumerate(line):
                if idx in [2, 3, 4, 5, 6]:
                    continue
                if idx > 6:
                    line[idx] = l2n_idx[idx-5][element]
                else:
                    line[idx] = l2n_idx[idx][element]
            for candidate in seleted:
                hole.append(list(candidate)) # 어디에 뚤렸는지 명시
                raw.append(line) # 새 데이터 추가

    shuffle_index = list(range(len(raw)))
    random.shuffle(shuffle_index)
    raw = np.array(raw)[shuffle_index] # 데이터 섞기
    hole = np.array(hole)[shuffle_index]
    print('data_len:', len(raw))
    np.save(newDataPath, {'data':raw, 'hole':hole})
    print('data save: {}'.format(newDataPath + '.npy'))

def generate_testDB(newDataPath=None, rawDataPath=None):
    if rawDataPath is None:
        rawDataPath = '../../data/test_kor.csv'
    if newDataPath is None:
        newDataPath = '../../data/samsung_test_db'

    l2n = get_label.get_labels()
    labels = ['주야', '요일', '발생지시도', '발생지시군구', '사고유형_대분류', '사고유형_중분류', '법규위반', '도로형태_대분류', '도로형태', '당사자종별_1당_대분류',
              '당사자종별_2당_대분류']
    l2n_idx = [l2n[label] for label in labels]
    with open(rawDataPath, newline='') as f:
        data = csv.reader(f, delimiter=',', quotechar='|')
        print('raw data loaded')
        raw = []
        hole = []
        for line_idx, line in enumerate(data):
            if line_idx == 0: # 타이틀 생략
                continue
            if line_idx % 1000 == 0:
                print('processing... {}/{}'.format(line_idx+1, 25001))

            candidate = []
            for i in [2, 3, 4, 5, 6]:
                if line[i] == '':
                    line[i] = -1
                    candidate.append(i)
                else:
                    line[i] = int(line[i])
            for idx, element in enumerate(line):
                if idx in [2, 3, 4, 5, 6]:
                    continue
                if idx > 6:
                    if element == '':
                        line[idx] = -1
                        candidate.append(idx)
                    else:
                        line[idx] = l2n_idx[idx-5][element]
                else:
                    if element == '':
                        line[idx] = -1
                        candidate.append(idx)
                    else:
                        line[idx] = l2n_idx[idx][element]
            hole.append(list(candidate)) # 어디에 뚤렸는지 명시
            raw.append(line) # 데이터 추가
    print('data_len:', len(raw))
    np.save(newDataPath, {'data': raw, 'hole':hole})
    print('data save: {}'.format(newDataPath + '.npy'))


if __name__ == '__main__':
    generate_DB()
    generate_testDB()