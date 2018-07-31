from collections import defaultdict
import csv

with open('Train_refine.csv',mode='r') as f:
	reader = csv.DictReader(f)
	columns = defaultdict(dict)
	for i, row in enumerate(reader):
		for (k,v) in row.items():
			if v!='':
				columns[k][v]=i
	columns['당사자종별_2당_대분류']['']=0
	columns['당사자종별_2당_대분류']['없음']=0
	print(columns)

