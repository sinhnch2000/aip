import torch
import re
import os

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from accelerate import PartialState

def generate_fn(examples, model, tokenizer):
    tokenized = tokenizer(examples['inputs'], padding='longest', return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **tokenized.to(model.device),
            num_return_sequences=1,
            num_beams=5,
        )
        results = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return {'predict' : results}


def reconstruct_fn(examples):
    def mapping_sample(examples):
        inputs, labels = [], []
        for idx in range(len(examples['instruction'])):
            item = examples['instruction'][idx] \
                .replace('{list_user_action}', examples['list_user_action'][idx].strip()) \
                .replace('{history}', examples['history'][idx].strip()) \
                .replace('{current}', examples['current'][idx].strip()) \
                .replace('{ontology}', examples['ontology'][idx].strip()) \

            inputs.append(re.sub('\s+', ' ', item))
            labels.append(re.sub('\s+', ' ', examples['label'][idx].strip()))
        return inputs, labels

    inputs, labels = mapping_sample(examples)

    return {'inputs': inputs, 'labels':labels}

def process(datasets, model, tokenizer, batch_size, output_path):

    datasets = datasets.map(
        lambda examples: reconstruct_fn(examples),
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=['list_user_action', 'history', 'current', 'ontology',
                        'instruction', 'id_turn', 'id_dialogue', 'label'],
        desc='Reconstruct input model'
    )

    datasets = datasets.map(
        lambda examples : generate_fn(examples, model, tokenizer),
        batched=True, batch_size=batch_size,
        desc='Predict processing'
    )

    datasets.to_json(output_path, force_ascii=False)

    return datasets


if __name__ == '__main__':
    ############ setup config ###########
    model_name_or_path = 'google/flan-t5-base'
    ckpt = 'checkpoint/pytorch_model_epoch3.bin'
    datapath = ['GradSearch_v2/test/fusedchat_test.json', 'GradSearch_v2/test/sgd_test.json']
    batch_size = 5
    output_path = 'predict'
    os.makedirs(output_path, exist_ok=True)
    ######################################

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model.load_state_dict(torch.load(ckpt))

    distributed_state = PartialState()
    model.to(distributed_state.device)

    with distributed_state.split_between_processes(datapath) as prompt:
        # for inference
        print("Prompt: ", prompt)
        for path in prompt:
            name_path = path.split('/')[-1]
            print(f"\n\n===================== Device {model.device}: {name_path} ===================")
            datasets = load_dataset('json', data_files=path, split='train')

            datasets = process(datasets, model, tokenizer, batch_size=batch_size, output_path=f'{output_path}/{name_path}')