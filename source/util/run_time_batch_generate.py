import csv
import itertools
import random

def generator_100lines_16C3(data, batch_size):
    _yield = []
    count = 0
    for line in data:
        temp_yield = []
        if count == batch_size:
            count = 0
            yield _yield
            _yield = []

        combs = list(itertools.combinations(line, 3))
        for comb in combs:
            temp_row = line[:]
            for j in range(3):
                temp_row[temp_row.index(comb[j])] = -1
            temp_yield.append(temp_row)

        sampled_yield = random.sample(temp_yield, int(len(temp_yield)*0.2))
        _yield = _yield + sampled_yield
        count += 1


read_f = False
write_f = False

read_f = open('16col_original_25000line_trainset.csv', newline='')
spamreader = csv.reader(read_f, delimiter=',', quotechar='|')


_generator = generator_100lines_16C3(spamreader, 100)
for data in _generator:
    print(len(data))


read_f.close()


    ### name = 16C3_data_number.csv / number는 001부터 251까지 예상

    #for i in range(1, int(25038 / 100) + 2):
    #     with open('16C3_data_' + str(i) + '.csv', 'w', newline='') as csvfile:
    #         spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #         spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    #         spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])