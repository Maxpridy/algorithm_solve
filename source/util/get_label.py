from collections import defaultdict
import csv
from pprint import pprint

def get_labels(filePath=None):
    if filePath == None:
        filePath = '../../data/Train_refine.csv'
    with open(filePath,mode='r') as f:
        reader = csv.DictReader(f)
        columns = defaultdict(dict)
        for i, row in enumerate(reader):
            for (k,v) in row.items():
                if v!='':
                    columns[k][v]=i
        columns['당사자종별_2당_대분류']['']=0
        columns['당사자종별_2당_대분류']['0']=0

    #pprint(columns)
    return columns

def get_n2c(filePath):
    c2n = get_labels(filePath)
    n2c = []
    for rkey, rvalue in c2n.items():
        sub_dict = {}
        for skey, svalue in rvalue.items():
            sub_dict[svalue] = skey
        if rkey == '당사자종별_2당_대분류':
            sub_dict[0] = '없음'
        n2c.append(sub_dict)
    return n2c

if __name__ == '__main__':
    n2c = get_n2c()
    pprint(n2c)