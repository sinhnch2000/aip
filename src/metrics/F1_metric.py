predict_test_path = "predict/fusedchat_test.json"
test = list(open(predict_test_path))
test = [p.replace('\n','') for p in test]
list_label_type=[]
list_predict_type=[]
count_tod = 0
count_odd = 0
for p in test:
    label_predict = p.split("predict")
    if "(current state)" in label_predict[1] and "labels" in label_predict[0]:
        label = label_predict[0].split("labels")[1][3:-3]
        label_type = label.split("type")[1][2:].split("current action")[0][:-2].strip()
        list_label_type.append(label_type)
        if "tod" == label_type:
          count_tod+=1
        else:
          count_odd+=1
        predict = label_predict[1][3:-2]
        predict_type = predict.split("type")[1][2:].split("current action")[0][:-2].strip()
        list_predict_type.append(predict_type)

right=0
total=0
for i in range(len(list_label_type)):
    if list_label_type[i] == list_predict_type[i]:
        right+=1
    total+=1
print("F1:", (right/total)*100)
print(len(test))
print(count_tod)
print(count_odd)
print(count_tod/count_odd)