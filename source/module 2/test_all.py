from DM import Dialogue_Manager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

dm = Dialogue_Manager()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model_dst = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
model_res = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
model_dst.load_state_dict(torch.load("C:\ALL\OJT\server\gradient_server_test\models\pytorch_model_dst_new.bin"))
model_res.load_state_dict(torch.load("C:\ALL\OJT\server\gradient_server_test\models\pytorch_model_res_new.bin"))
embedding_size_dst = model_dst.get_input_embeddings().weight.shape[0]
embedding_size_res = model_res.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size_dst:
    model_dst.resize_token_embeddings(len(tokenizer))
if len(tokenizer) > embedding_size_res:
    model_res.resize_token_embeddings(len(tokenizer))

sample_dst = {
    "instruction": "The goal of this assignment is to determine the belief state by analyzing the dialogue. Giving the list of personal action [ACTIONS] {list_user_action}. When user query is out of the ontology, respond \"Unsure about answer, you should find with SearchEngine [TERM]\" where TERM is the search term you want to find out if not sure about the answer. Input: <CTX> {context} <QUERY> {current_query} <ONTOLOGY> {ontology}. Output: ",
    "list_user_action": "inform, request, inform_intent, negate_intent, affirm_intent, affirm, negate, select, thank_you, goodbye, greet, general_asking, request_alts",
    "ontology": "HOTELS_1:(slot0=location of the hotel; slot1=number of rooms in the reservation; slot2=start date for the reservation; slot3=number of days in the reservation; slot4=star rating of the hotel; slot5=name of the hotel; slot6=address of the hotel; slot7=phone number of the hotel; slot8=price per night for the reservation; slot9=boolean flag indicating if the hotel has wifi)",
    "current_query": "",
    "context": ""
}
sample_res = {
    "instruction": "Follow the conversation context of the task is taken into consideration <CTX> {context} <EOD> and ensure that use provides in the given <K> {ontology} {system_action} {documents} you must respond to the conversation with <S> {style}. <Q> What should be taken to complete the task effectively?",
    "context": "",
    "ontology": "HOTELS_1:(slot0=location of the hotel; slot1=number of rooms in the reservation; slot2=start date for the reservation; slot3=number of days in the reservation; slot4=star rating of the hotel; slot5=name of the hotel; slot6=address of the hotel; slot7=phone number of the hotel; slot8=price per night for the reservation; slot9=boolean flag indicating if the hotel has wifi)",
    "system_action": "",
    "documents": "",
    "style": "politely"
}

context = []
def add_utterance(utterance, size):
    context.append(utterance)
    if len(context) > size:
        context.pop(0)

while(1):
    # print("\nContext:", ((" ").join(utt for utt in context)).strip())
    current_query = input("USER:  ")
    if current_query == 'exit':
        break
    sample_dst["current_query"] = "USER: " + current_query
    item_dst = sample_dst['instruction'] \
            .replace('{list_user_action}', sample_dst['list_user_action'].strip()) \
            .replace('{context}', sample_dst['context'].strip()) \
            .replace('{current_query}', sample_dst['current_query'].strip()) \
            .replace('{ontology}', sample_dst['ontology'].strip())

    input_tokens = tokenizer(item_dst, return_tensors="pt")
    output = model_dst.generate(input_tokens["input_ids"], attention_mask=input_tokens["attention_mask"], max_new_tokens=100)
    data_dst_1 = tokenizer.decode(output[0], skip_special_tokens=True)

    dm.format_string_input(data_dst_1)
    if dm.classify_dst == "tod":
        dm.transform_action()

        sample_res["system_action"] = dm.format_string_output()


    add_utterance("USER: " + current_query, 5)
    sample_dst["context"] = ((" ").join(utt for utt in context)).strip()

    item_res = sample_res['instruction'] \
        .replace('{ontology}', sample_res['ontology'].strip()) \
        .replace('{context}', sample_res['context'].strip()) \
        .replace('{style}', sample_res['style'].strip()) \
        .replace('{system_action}', sample_res['system_action'].strip()) \
        .replace('{documents}', sample_res['documents'].strip())

    input_tokens = tokenizer(item_res, return_tensors="pt")
    output = model_res.generate(input_tokens["input_ids"], attention_mask=input_tokens["attention_mask"],
                                max_new_tokens=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("AGENT: " + "\033[91m" + response + "\033[0m")
    if dm.classify_dst == "tod":
        print("---action:", dm.policy.system_action)
    add_utterance("AGENT: " + response, 5)
    if "GOODBYE" in dm.policy.system_action:
        break


