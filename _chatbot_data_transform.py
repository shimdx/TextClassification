import ast
import random

data = []
with open('data/bot_dataset_all.txt', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(ast.literal_eval(line))

with open('data/bot_dataset_test.txt', 'w', encoding='utf-8') as f_test:
    with open('data/bot_dataset_train.txt', 'w', encoding='utf-8') as f_train:
        random.shuffle(data)
        testcnt = {"life":0,"svc":0,"shp":0,"tour":0}
        traincnt = {"life":0,"svc":0,"shp":0,"tour":0}
        # testcnt = {"life": 0, "bot": 0, "shp": 0, "tour": 0, "dialog": 0}
        for line in data:
            if testcnt[line[1]] < 250:
                f_test.write("('" + line[0].replace(","," ").replace("'","")  + "', '" + line[1] + "')\n")
                testcnt[line[1]] += 1
            elif traincnt[line[1]] < 700:
                f_train.write("('" + line[0].replace(","," ").replace("'","") + "', '" + line[1] + "')\n")
                traincnt[line[1]] += 1


random.shuffle(data)
# {'life': 0, 'tour': 1, 'svc': 2, 'shp': 3}
label = {"life":0,"svc":1,"shp":2,"tour":3}
# label = {"life": 0, "bot": 1, "shp": 2, "tour": 3, "dialog": -1}
with open('data/bot_dataset_test.csv', 'w', encoding='utf-8') as f_test:
    f_test.write("label,content\n")
    with open('data/bot_dataset_train.csv', 'w', encoding='utf-8') as f_train:
        f_train.write("label,content\n")
        testcnt = {"life":0,"svc":0,"shp":0,"tour":0}
        traincnt = {"life":0,"svc":0,"shp":0,"tour":0}
        for line in data:
            if testcnt[line[1]] < 250:
                f_test.write(str(label[line[1]])+","+line[0].replace(","," ")+"\n")
                testcnt[line[1]] += 1
            elif traincnt[line[1]] < 700:
                f_train.write(str(label[line[1]])+","+line[0].replace(","," ")+"\n")
                traincnt[line[1]] += 1
