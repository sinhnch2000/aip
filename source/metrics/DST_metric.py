from datasets import load_dataset

predict_test_path = r"C:\ALL\GRADIENT\SERVER\gradient_server_test\predict\fusedchat_test.json"
test = load_dataset(predict_test_path)
print(test)
test = [p.replace('\n','') for p in test]

list_label_state=[]
list_predict_state=[]
for p in test:
    p = p.replace('\n','')
    label_predict = p.split("predict")

    label = label_predict[0].split("labels")[1][3:-3]
    label_state = label.split("(current state)")[1].strip()
    list_label_state.append(label_state)

    predict = label_predict[1][3:-2]
    predict_state = predict.split("(current state)")[1].strip()
    list_predict_state.append(predict_state)

def get_list_state(state):
    list_state = state.split("||")
    for i in range(len(list_state)):
        list_state[i] = list_state[i].strip()
    return set(list_state)

wrong=0
total=0
for i in range(len(list_label_state)):
    list_gold_state = get_list_state(list_label_state[i])
    list_dict_state = get_list_state(list_predict_state[i])
    if list_gold_state != list_dict_state:
        print("list_gold_state:",list_gold_state)
        print("list_dict_state:",list_dict_state)
        print("bug_gold_state:", list_gold_state-list_dict_state)
        print("bug_dict_state:", list_dict_state-list_gold_state)
        wrong+= 1
        print("\n")
    total+=1
print("JGA:", 1-wrong/total)