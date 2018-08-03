import csv
import itertools
import random
import numpy as np
import os
import json
from source.util import get_label

def generate_DB(newDataPath=None, rawDataPath=None, sample_for_line=10, hole_value=-1):
    if rawDataPath is None:
        rawDataPath = '../../data/16col_original_25000line_trainset.csv'
    if newDataPath is None:
        newDataPath = '../../data/samsung_db.json'

    l2n = get_label.get_labels()
    labels = ['주야', '요일', '발생지시도', '발생지시군구', '사고유형_대분류', '사고유형_중분류', '법규위반', '도로형태_대분류', '도로형태', '당사자종별_1당_대분류',
              '당사자종별_2당_대분류']
    l2n_idx = [l2n[label] for label in labels]
    with open(rawDataPath, newline='') as f:
        data = csv.reader(f, delimiter=',', quotechar='|')
        print('raw data loaded')

        DB = []
        for line_idx, line in enumerate(data):
            if line_idx % 1000 == 0:
                print('processing... {}/{}'.format(line_idx+1, 25001))
            candidates = itertools.combinations(range(len(line)), 3) # 3개 조합 계산
            seleted = random.sample(list(candidates), sample_for_line) # 모든 경우의 수 중 X개만 선택
            #print('원본', line)
            for i in [2, 3, 4, 5, 6]:
                line[i] = int(line[i])
            for idx, element in enumerate(line):
                if idx in [2, 3, 4, 5, 6]:
                    continue
                if idx > 6:
                    line[idx] = l2n_idx[idx-5][element]
                else:
                    line[idx] = l2n_idx[idx][element]
            #print('처리', line)
            for candidate in seleted:
                temp_line = line.copy()
                for hole in candidate:
                    temp_line[hole] = hole_value # -1로 채움

                temp_line.append(list(candidate)) # 어디에 뚤렸는지 명시
                DB.append(temp_line) # 새 데이터 추가
                #print('뚫뚫', temp_line)


    #DB = np.array(DB)
    print('created data#: {}'.format(len(DB)))
    with open(newDataPath, 'w') as f:
        json.dump(DB, f, indent='\t')
    print('save data')

if __name__ == '__main__':
    generate_DB()