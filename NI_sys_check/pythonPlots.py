import matplotlib.pyplot as plt

temp = list()
items = list()
with open('ValeroLabTesting/testandMest_1_3.csv') as csv_file:
    for row in csv_file:
        temp = row.split(", ")
        if(len(temp)>len(items)):
            for i in range(len(items), len(temp)):
                items.append(list())
        for i in range(len(temp)):
            items[i].append(temp[i])
for i in range(len(items)):
    for j in range(len(items[i])):
        if i == 0:
            items[i][j] = round(float(items[i][j]), 3)
        else:
            items[i][j] = round(float(items[i][j]), 5)

'''
for i in items:
    temp = "["
    for p in i:
        temp = temp + str(p) + " "
    temp += "]"
    print(temp)
'''

line_chart1 = plt.plot(items[0], items[1], '-')
line_chart2 = plt.plot(items[0], items[2], '-')
line_chart3 = plt.plot(items[0], items[3], '-')

plt.show()
