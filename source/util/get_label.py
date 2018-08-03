from collections import defaultdict
import csv

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
		columns['당사자종별_2당_대분류']['없음']=0
		# print(columns)
		return columns


if __name__ == '__main__':
	label = get_labels()
	label