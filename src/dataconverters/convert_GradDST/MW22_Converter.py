import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks/baseline_v1/gradients.baselinev1.dialogstate')

import json
import random
import glob
import os
import numpy as np

from pathlib import Path
from typing import List, Dict, Tuple, Type, Union, Optional
from sklearn.model_selection import train_test_split

from converter import DialConverter


WOZ_DMS = ['taxi', 'police', 'hospital', 'hotel',
            'attraction', 'train', 'restaurant', 'bus']


class MW22_Converter(DialConverter):
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
        data_path_list = glob.glob(os.path.join(self.file_path,'*.json'))
        for data_path in data_path_list:
            filename = os.path.basename(data_path)
            dataset = self.load_datapath(data_path)
            
            list_instructions = self.define_instruct(instruct_path)
            list_ontologies = self.define_ontology(ontolopy_path)
            samples = []
            # Analyze all dialogues
            for dialogue in dataset:
                # get summarize dialogue
                gold_domain, domain_name = self.get_gold_dm(dialogue, list_ontologies)
                # process dialogue into sub-dialogue
                cw = self.window_context - 1
                
                for idx in range(len(dialogue['turns'])):
                    sub_dialogue = []
                    if idx % 2 == 0:
                        sub_dialogue.append(dialogue['turns'][max(0, idx - cw):max(0, idx + 1)])
                        #print(key, sub_dialogue)
                        sample = self.get_sample(sub_dialogue, gold_domain,
                                                list_ontologies, list_instructions)
                        samples.append(sample)

            self.save_datapath(samples, filename)

    
    def get_sample(self, sub_dialogue, gold_dm, list_ontologies, list_instructions):
        for _, child in enumerate(sub_dialogue):
            # get context
            item = dict()
            ls_turn = []
            for turn in child:
                speaker = 'USER: ' if turn['speaker'] == 'USER' else 'SYSTEM: '
                ls_turn.append(speaker + turn['utterance'])

            if len(ls_turn) == 1:
                item['context'] = ''
                item['current_query'] = ls_turn[0]
            else:
                item['context'] = ' '.join([ls_turn[idx].strip() for idx in range(len(ls_turn)-1)])
                item['current_query'] = ls_turn[-1][6:]
            item['instruction'] = self.get_instruction(list_instructions)
            item['list_user_action'] = ', '.join(act.lower() for act in self.useracts)

            # system_action - ontology
            ls_useracts, ls_domains = [], ['']
            current_state = child[-1]['dialog_act'].items()
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
                        idx_name = domain.lower()
                        value_onto, onto_mapping = self.get_ontology(idx_name, list_ontologies)
                        item['ontology'] = value_onto

                        ls_txt = []
                        for value in frames:
                            if value[0] in ['choice', 'none']:
                                ls_txt.append(value[1])
                            else:
                                try:
                                    ls_txt.append(onto_mapping[value[0]] + '=' + value[1])
                                except:
                                    ls_txt.append(value[0])

                        acts = [action.lower() + '(' + uttr + ')' for uttr in ls_txt]
                        temp_acts = ' and '.join(act for act in acts)
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

            if len(ls_domains) == 1:
                item['label'] = 'and '.join(item for item in ls_useracts)
            else:
                item['label'] = ls_domains[-1].upper() + ':[' + 'and '.join(item for item in ls_useracts) + ']'
        return item
        
    def load_datapath(self, data_path) -> List[Dict]:
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
        map_ontology_domain = {}
        for slot in ontologies[domain].keys():
            map_ontology_domain.setdefault(slot, "slot" + str(count))
            count = count + 1
        return map_ontology_domain

    def get_gold_dm(self, dialogue, ontologies):
        gold_dm, dm_name = [], []
        for dm in dialogue['services']:
            if dm in WOZ_DMS:
                onto_mapping = self.map_ontology(ontologies, dm)
                tmps = [onto_mapping[k]  + "=" + v 
                        for k, v in ontologies[dm].items()]
                value_onto = dm.upper() + ":(" + '; '.join(tmp for tmp in tmps) + ")"
                gold_dm.append(value_onto)
                dm_name.append(dm)
                    
        return gold_dm, dm_name

    def get_ontology(self, domain_name, ontologies):
        onto_mapping = self.map_ontology(ontologies, domain_name)
        tmps = [onto_mapping[key]  + "=" + value 
            for key, value in ontologies[domain_name].items()]
        value_onto = domain_name.upper() + ":(" + '; '.join(tmp for tmp in tmps) + ")"

        return value_onto, onto_mapping

    def save_datapath(self, data_processed: List[Dict], filename: str):
        with open(os.path.join(self.save_path, filename), 'w') as f:
            json.dump(data_processed, f, indent=4)

    def get_instruction(self, list_instructions):
        random_instruction = list_instructions[random.randint(0, len(list_instructions) - 1)]
        return random_instruction


if __name__ == '__main__':
    # TEST
    fusedchat_converter = MW22_Converter(
        file_path=r'C:\ALL\OJT\SERVER\gradient_server_test\data\raw\MULTIWOZ\MW 2.2\pre_process',
        save_path=r'C:\ALL\OJT\SERVER\gradient_server_test\data\interim\GradSearch\MW22',
        window_context=5).__call__(
        instruct_path=r"C:\Users\This PC\Downloads\instruct_GradSearch.txt",
        ontolopy_path=r"C:\ALL\OJT\SERVER\gradient_server_test\data\processed_schema\schema_final.json")







