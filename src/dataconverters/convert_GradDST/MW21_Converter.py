import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks/baseline_v1/gradients.baselinev1.dialogstate')

import json
import random
import glob
import os
from typing import List, Dict, Union, Optional
from src.features.converter import DialConverter

WOZ_DMS = ['taxi', 'hospital', 'hotel', 'attraction', 'train', 'restaurant', 'police']

class MW21_Converter(DialConverter):
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
        self.useracts = ['INFORM', 'THANK_YOU', 'REQUEST', 'GREET', 'GOODBYE']


    def __call__(self, instruction_path, ontolopy_path):
        print(f"Start  processing {self.__class__.__name__}")
        self.process(instruction_path=instruction_path, ontolopy_path=ontolopy_path)
        print(f"Finish processing {self.__class__.__name__} at {self.save_path}")

    def process(self, instruction_path, ontolopy_path):
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
            list_instruction = self.load_instruction(instruction_path)
            list_ontology = self.load_ontology(ontolopy_path)

            all_samples = []
            # Analyze all dialogues
            for dialogue in dataset.values():
                # get summarize dialogue
                gold_domain, domain_name = self.get_gold_dm(dialogue, list_ontology)
                # process dialogue into sub-dialogue
                sub_dialogue = []
                for idx in range(len(dialogue['log'])):
                    if idx % 2 == 0:
                        sub_dialogue.append(dialogue['log'][max(0, idx - cw):max(0, idx + 1)])
                samples = self.get_samples(sub_dialogue, gold_domain, domain_name, list_ontology, list_instruction)
                set.extend(samples)

            self.save_datapath(set, filename)

    def get_samples(self, sub_dialogue, gold_dm, dm_name, list_ontologies, list_instructions):
        samples = []
        for _, child in enumerate(sub_dialogue):

            # get context
            item = dict()
            ls_turn = []
            for turn in child:
                speaker = 'USER: ' if turn['speaker'] == 'USER' else 'SYSTEM: '
                ls_turn.append(speaker + turn['text'])

            if len(ls_turn) == 1:
                item['context'] = ''
                item['current_query'] = ls_turn[0][6:]
            else:
                item['context'] = ' '.join([ls_turn[idx].strip() for idx in range(len(ls_turn)-1)])
                item['current_query'] = ls_turn[-1][6:]
            item['instruction'] = self.get_instruction(list_instructions)
            item['list_user_action'] = ', '.join(act.lower() for act in self.useracts)

            # system_action - ontology
            ls_useracts, ls_domains = [], []
            if "dialog_act" not in child[-1].keys():
                item['ontology'] = ' || '.join(idx for idx in gold_dm)
            else:
                if child[-1]['dialog_act'] == {}:
                    item['ontology'] = ' || '.join(idx for idx in gold_dm)
                else:
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
                            # item['ontology'] = DOMAIN:(slot0=des0,slot1=des1,slot2=des2)

                            ls_txt = []
                            for value in frames:
                                slot = value[0].lower()
                                if slot in ['choice', 'none']:
                                    continue
                                else:
                                    try:
                                        # onto_mapping =
                                        # {  "slot0":{"area to search for attractions": ["area"]},
                                        #    "slot1":{"name of the attraction": ["name"]},
                                        #    "slot2":{"type of the attraction": ["type"]}}
                                        for slotstr, description_listslots in onto_mapping.items():
                                            if slot in list(description_listslots.values())[0]:
                                                ls_txt.append(slotstr + '=' + value[1])
                                    except:
                                        ls_txt.append(slot)

                            acts = [action.lower() + '(' + uttr + ')' for uttr in ls_txt]
                            temp_acts = ' and '.join(act for act in acts)
                            ls_useracts.append(temp_acts.replace('(none)', '').replace('=?', ''))

                        else:  # (general-thank | general-bye | general-greet) just a state not a domain
                            item['ontology'] = ' || '.join(idx for idx in gold_dm)
                            if action.lower() == 'greet':
                                ls_useracts.append('chitchat')
                            else:
                                ls_useracts.append(action.lower() \
                                                   .replace('thank', 'thank_you') \
                                                   .replace('bye', 'goodbye'))

            for i in ls_useracts:
                if len(i) == 0 or i == '':
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
            samples.append(item)
        return samples

    # LOAD and SAVE data
    def load_datapath(self, data_path) -> List[Dict]:
        with open(data_path, 'r+') as f:
            dataset = json.load(f)
        return dataset

    def save_datapath(self, data_processed: List[Dict], filename: str):
        with open(os.path.join(self.save_path, filename), 'w') as f:
            json.dump(data_processed, f, indent=4)

    # LOAD and GET instruction
    def load_instruction(self, instruct_path) -> List[str]:
        with open(instruct_path, encoding="utf8") as f:
            instructions = f.readlines()
        return instructions

    def get_instruction(self, list_instructions):
        random_instruction = list_instructions[random.randint(0, len(list_instructions) - 1)]
        return random_instruction

    # LOAD and MAP and GET ontology
    def load_ontology(self, ontolopy_path: Optional[str] = None) -> Union[Dict, List[Dict]]:
        with open(ontolopy_path, encoding="utf8") as f:
            ontologies = json.load(f)
        return ontologies

    def map_ontology(self, ontologies, domain, count=0):
        map_ontology_domain = {}
        for description, lists_slot in ontologies[domain.lower()].items():
            map_ontology_domain.setdefault("slot" + str(count), {description:lists_slot})
            count = count + 1
        return map_ontology_domain
        # {  "slot0":{"area to search for attractions": ["area"]},
        #    "slot1":{"name of the attraction": ["name"],
        #    "slot2":{"type of the attraction": ["type"]}}

    def get_ontology(self, domain_name, ontologies):
        onto_mapping = self.map_ontology(ontologies, domain_name)
        tmps = []
        for slotstr, description_listslots in onto_mapping.items():
            tmps.append(slotstr+"="+list(description_listslots.keys())[0])

        value_onto = domain_name.upper() + ":(" + '; '.join(tmp for tmp in tmps) + ")"
        return value_onto, onto_mapping
        # value_onto = DOMAIN:(slot0=des0,slot1=des1,slot2=des2)

    def get_gold_dm(self, dialogue, ontologies):
        gold_dm, dm_name = [], []
        tmp_dm = set()
        goal = dialogue["goal"]
        for domain, all_info in goal.items():
            if domain not in ["message", "topic"] and all_info != {}:
                if "info" in all_info.keys():
                    tmp_dm.add(domain)
        for dm in tmp_dm:
            if dm.lower() in WOZ_DMS:
                value_onto, onto_mapping = self.get_ontology(dm, ontologies)
                gold_dm.append(value_onto)
                dm_name.append(dm)
        return gold_dm, dm_name

if __name__ == '__main__':
    # TEST
    fusedchat_converter = MW21_Converter(
        file_path=r'C:\ALL\OJT\SERVER\gradient_server_test\data\data raw\MW21',
        save_path=r'C:\ALL\OJT\SERVER\gradient_server_test\data\data interim\GradSearch\MW21',
        window_context=5).__call__(
        instruct_path=r"C:\ALL\OJT\SERVER\gradient_server_test\data\instructions\instruct_GradSearch.txt",
        ontolopy_path=r"C:\ALL\OJT\SERVER\gradient_server_test\data\schema guided\schema.json")







