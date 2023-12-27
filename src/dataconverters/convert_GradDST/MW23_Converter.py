import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks/baseline_v1/gradients.baselinev1.dialogstate')

import json
import random
import glob
import os
from typing import List, Dict, Tuple, Type, Union, Optional, Any
from src.features.converter import DialConverter


WOZ_DMS = ['taxi', 'hospital', 'hotel', 'attraction', 'train', 'restaurant', 'bus']

class MW23_Converter(DialConverter):
    def __init__(self,
                 file_path: str,
                 save_path: str,
                 tag_speaker: str = 'USER',
                 tag_agent: str = 'AGENT',
                 window_context: int = 0,
                 ) -> None:
        """
        Args:
            save_path: path to save the processed dataset
        """
        super().__init__(file_path,
                         save_path,
                         tag_speaker,
                         tag_agent,
                         window_context,
                         )
        self.useracts = ['INFORM', 'THANK', 'REQUEST', 'GREET', 'BYE']

    def __call__(self, instruct_path, ontolopy_path):
        print(f"Start  processing {self.__class__.__name__}")
        self.process(instruct_path=instruct_path, ontolopy_path=ontolopy_path)
        print(f"Finish processing {self.__class__.__name__} at {self.save_path}")

    def process_schema(self, ontology) -> Dict:
        new_ontology = {}
        for domain_slot, description in ontology.items():
            domain, slot = domain_slot.split("-")
            if domain not in new_ontology:
                new_ontology[domain] = {slot: description[0]}
            else:
                new_ontology[domain][slot] = description[0] # description is type list and only need first one
        return new_ontology

    def process(self, instruct_path, ontolopy_path):
        """Implement your convert logics in this function
            1. `instruction`: instruction for zero-shot learning
            2. `context`: means dialogue history
            3. `state_of_user`: a support document to response current
                user's utterance
            4. `system_action`: what system intends to do
            5. `response`: label for response generation
        A dataset after being formatted should be a List[Dict] and saved
        as a .json file
        Note: text preprocessing needs to be performed during transition
                from crawl dataset to module3-based format
        """
        # Separate complete dialogues to sub dialogues
        # data_path = os.path.join(self.file_path,'data.json')
        data_path_list = glob.glob(os.path.join(self.file_path,'*.json'))
        print(data_path_list)
        # val_path = os.path.join(self.file_path,'valListFile.txt')
        # test_path = os.path.join(self.file_path,'testListFile.txt')
        # dataset contains all dialogue, while valtest 
        # and testset contains only file name
        list_instructions = self.define_instruct(instruct_path)
        raw_ontology = self.define_ontology(ontolopy_path) # file to slot_description.json
        final_ontologies = self.process_schema(raw_ontology)

        for data_path in data_path_list:
            filename = os.path.basename(data_path)
            dataset = self.load_datapath(data_path)

            set = []
            # Analyze all dialogues
            for keys, dialogue in dataset.items():
                # get summarize dialogue
                gold_domain, domain_name = self.get_gold_dm(dialogue, final_ontologies)
                # process dialogue into sub-dialogue
                cw = self.window_context - 1
                
                sub_dialogue = []
                for idx in range(len(dialogue['log'])):
                    if idx % 2 == 0:
                        sub_dialogue.append(dialogue['log'][max(0, idx - cw):max(0, idx + 1)])
                        #print(key, sub_dialogue)
                processed_samples = self.get_samples(sub_dialogue, gold_domain, domain_name,
                                        final_ontologies, list_instructions)
                set.extend(processed_samples)

            self.save_datapath(set, filename)

    def get_samples(self, 
                   sub_dialogue: List[dict], 
                   gold_dm: List[Any], 
                   domain_name: List[Any], 
                   list_ontologies, 
                   list_instructions):
        samples = []
        for _, child in enumerate(sub_dialogue):
            # get context
            item = dict()
            ls_turn = []
            for idx, turn in enumerate(child):
                speaker = 'USER: ' if idx % 2 == 0 else 'SYSTEM: '
                ls_turn.append(speaker + turn['text'])

            if len(ls_turn) == 1:
                item['context'] = ''
                item['current_query'] = ls_turn[0][6:] # remove speaker
            else:
                item['context'] = ' '.join([ls_turn[idx].strip() for idx in range(len(ls_turn)-1)])
                item['current_query'] = ls_turn[-1][6:] # remove speaker
            item['instruction'] = self.get_instruction(list_instructions)
            item['list_user_action'] = ', '.join(act.lower() for act in self.useracts)

            ######
            # system_action - ontology
            ls_useracts, ls_domains = [], []
            current_state = child[-1]['dialog_act'].items() # get last turn
            # "dialog_act": {"Hotel-Inform": [["Stay","2"], ]}
            for domain_action, frames in current_state:
                """
                domain_action: domain | action
                frame: list[[slot, value], [slot, value]]
                """
                dm_act = domain_action.split('-')
                domain = dm_act[0]
                action = dm_act[1]
                if domain.lower() in WOZ_DMS:
                    ls_domains.append(domain)
                    domain = domain.lower()
                    value_onto, onto_mapping = self.get_ontology(domain, list_ontologies)
                    item['ontology'] = value_onto

                    ls_txt = []
                    # [["Stay","2"], ]
                    for value in frames:
                        slot = value[0].lower()
                        if slot in ['choice', 'none']:
                            # ls_txt.append(value[1])
                            continue
                        else:
                            try:
                                ls_txt.append(onto_mapping[slot] + '=' + value[1])
                            except:
                                ls_txt.append(slot)
                    
                    # [slot1=2, ...]
                    acts = [action.lower() + '(' + uttr + ')' for uttr in ls_txt]
                    # [inform(slot1=2), ...]
                    temp_acts = ' and '.join(act for act in acts)
                    # inform(slot1=2) and ...
                    ls_useracts.append(temp_acts.replace('(none)', '').replace('=?', ''))

                else: # (general-thank | general-bye | general-greet) just a state not a domain
                    item['ontology'] = ' || '.join(idx for idx in gold_dm)
                    if action.lower() == 'greet':
                        ls_useracts.append('chitchat')
                    else:
                        ls_useracts.append(action.lower() \
                                    .replace('thank', 'thank_you') \
                                    .replace('bye', 'goodbye'))
                        # try:
                        #     previous_state = list(child[-2]['dialog_act'].keys())
                        #     dm_name = (previous_state[0].split('-')[0]).lower()
                        #     ls_domains.append(dm_name)
                        # except:
                        #     try:
                        #         previous_state = list(child[-3]['dialog_act'].keys())
                        #         dm_name = (previous_state[0].split('-')[0]).lower()
                        #         if dm_name in domain_name:
                        #             ls_domains.append(dm_name)
                        #         else:
                        #             ls_domains.append(domain_name[0])
                        #     except:
                        #         print(domain_name)
                        #         ls_domains.append(domain_name[0])
            for i in ls_useracts:
                if len(i) == 0 or i =='':

                    ls_useracts.remove(i)

            if len(ls_domains) == 0:
                item['label'] = ' and '.join(item for item in ls_useracts)
            else:
                if len(ls_useracts) == 0:
                    item['label'] = ""
                else:
                    if ls_useracts[0] != "":
                        item['label'] = ls_domains[-1].upper() + ':[' + ' and '.join(item for item in ls_useracts) + ']'
                    else:
                        item['label'] = ""
            if "ontology" in item.keys():
                samples.append(item)
        return samples

    def load_datapath(self, data_path):
        with open(data_path, 'r+') as f:
            dataset = json.load(f)
        return dataset

    def define_instruct(self, instruct_path) -> List[str]:
        with open(instruct_path, encoding="utf8") as f:
            instructions = f.readlines()
        return instructions

    def define_ontology(self, ontolopy_path: Optional[str] = None) -> Union[Dict, List[Dict]]:
        with open(ontolopy_path, encoding="utf8") as f:
            ontologies = json.load(f)
        return ontologies    
    
    def map_ontology(self, ontologies, domain, count=0):
        # label slot0 to slotN for matched domain
        map_ontology_domain = {}
        for slot in ontologies[domain].keys():
            map_ontology_domain.setdefault(slot, "slot" + str(count)) # {slot: slot0}
            count = count + 1
        return map_ontology_domain

    def get_gold_dm(self, 
                    dialogue, 
                    ontologies) -> Tuple[List, List]:
        # ontology <- slot_descriptions.json
        gold_dm, dm_name = [], []
        tmp_dm = set()
        for log in dialogue["log"]:
            if "dialog_act" in log.keys():
                dialog_act = log["dialog_act"]
                for domain_action in dialog_act.keys():
                    domain = domain_action.split("-")[0]
                    tmp_dm.add(domain)

        for dm in tmp_dm:
            if dm in WOZ_DMS:
                onto_mapping = self.map_ontology(ontologies, dm)
                tmps = [onto_mapping[k]  + "=" + v #slot0 = description
                        for k, v in ontologies[dm].items()]
                value_onto = dm.upper() + ":(" + '; '.join(tmp for tmp in tmps) + ")"
                gold_dm.append(value_onto)
                dm_name.append(dm)
        # return list of ontology of matched domain and name            
        return gold_dm, dm_name
    
    def get_ontology(self, domain_name, ontologies):
        onto_mapping = self.map_ontology(ontologies, domain_name)
        tmps = [onto_mapping[key.lower()]  + "=" + value.lower()
            for key, value in ontologies[domain_name].items()]
        value_onto = domain_name.upper() + ":(" + '; '.join(tmp for tmp in tmps) + ")"

        return value_onto, onto_mapping

    def save_datapath(self, data_processed: List[Dict], filename: str):
        with open(os.path.join(self.save_path, filename), 'w+') as f:
            json.dump(data_processed, f, indent=4)

    def get_instruction(self, list_instructions):
        random_instruction = list_instructions[random.randint(0, len(list_instructions) - 1)]
        return random_instruction
    
if __name__ == '__main__':
    # TEST
    MW23_converter = MW23_Converter(
        file_path=r'D:\Work\baseline_v1\multiWOZ\get_data_multiwoz\MultiWOZ2_3\raw',
        save_path=r'D:\Work\baseline_v1\multiWOZ\get_data_multiwoz\MultiWOZ2_3\interim',
        window_context=5).__call__(
        instruct_path=r"D:\Work\baseline_v1\multiWOZ\get_data_multiwoz\MultiWOZ2_3\instruct_GradSearch.txt",
        ontolopy_path=r"D:\Work\baseline_v1\multiWOZ\get_data_multiwoz\MultiWOZ2_3\slot_descriptions.json")

    