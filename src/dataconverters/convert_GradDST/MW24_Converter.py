import sys
import json
import random
import glob
import os
from typing import List, Dict, Union, Optional
from src.dataconverters.converter import DialConverter

sys.path.append('/content/drive/MyDrive/Colab Notebooks/baseline_v1/gradients.baselinev1.dialogstate')
list_woz_domain = ['taxi', 'police', 'hospital', 'hotel', 'attraction', 'train', 'restaurant']
list_user_action = ['inform', 'request', 'inform_intent', 'negate_intent', 'affirm_intent', 'affirm', 'negate', 'select', 'thank_you', 'goodbye', 'greet', 'general_asking', 'request_alts']

class MW24Converter(DialConverter):
    def __init__(self,
                 file_path: str,
                 save_path: str,
                 tag_user: str = 'USER',
                 tag_system: str = 'SYSTEM',
                 window_context: int = 3  # slide window
                 ) -> None:
        """
        Args:
            save_path: path to save the processed dataset
        """
        super().__init__(file_path, save_path, tag_user, tag_system)
        self.window_context = window_context

    def __call__(self, instruction_path, ontolopy_path):
        print(f"Start  processing {self.__class__.__name__}")
        self.process(instruction_path=instruction_path, ontolopy_path=ontolopy_path)
        print(f"Finish processing {self.__class__.__name__} at {self.save_path}")

    def process(self, instruction_path, ontolopy_path):
        # Separate complete dialogues to sub dialogues
        list_data_path = glob.glob(os.path.join(self.file_path,'*.json'))
        list_instruction = self.load_instruction(instruction_path)
        list_ontology = self.load_ontology(ontolopy_path)

        for data_path in list_data_path:
            filename = os.path.basename(data_path)
            dataset = self.load_datapath(data_path)

            list_all_sample = []
            cw = self.window_context - 1  # slide window
            # Analyze all dialogues
            for dialogue in dataset:
                id = dialogue["dialogue_idx"]
                # get summarize dialogue
                # process dialogue into sub-dialogues
                list_sub_dialogue = []
                list_turns = dialogue['dialogue']
                for i in range(len(list_turns)):
                    child_dialogue = list_turns[max(0, i - cw):max(0, i + 1)]
                    list_sub_dialogue.append(child_dialogue)
                # process raw list_sub_dialogue to interim list_sub_sample
                list_sub_sample = self.get_list_sub_sample(id, dialogue["domains"], list_sub_dialogue, list_ontology, list_instruction)
                list_all_sample.extend(list_sub_sample)
            self.save_datapath(list_all_sample, filename)

    def get_list_sub_sample(self, id, all_domains, list_sub_dialogue, list_ontology, list_instruction):
        list_sub_sample = []
        previous_state = dict()
        for _, sub_dialogue in enumerate(list_sub_dialogue):
            item = dict()
            list_current_action = list()
            dict_action_one_turn = dict()

            # get context, current_user, instruction, list_user_action and ontology
            list_turn = []
            list_turn.append("USER: " + sub_dialogue[0]["transcript"])
            for turn in sub_dialogue[1:]:
                if turn["system_transcript"] != '':
                    list_turn.append("AGENT: " + turn["system_transcript"])
                list_turn.append("USER: " + turn["transcript"])

            item['instruction'] = self.get_instruction(list_instruction).strip()
            item['list_user_action'] = ', '.join(action.lower() for action in list_user_action)
            item['ontology'] = ''
            item['id_dialogue'] = id
            item['id_turn'] = sub_dialogue[-1]['turn_idx']
            item['context'] = ' '.join([list_turn[i].strip() for i in range(len(list_turn)-1)]).strip()
            item['current_query'] = list_turn[-1].strip()
            if item['context'] != "":
                item['current_query'] = item['current_query'][6:]

            # get type, current_state and current_action
            belief_state = sub_dialogue[-1]['belief_state']

            if len(belief_state) == 0: # ODD || "dialog_act": {},
                if "general" not in dict_action_one_turn.keys():
                    dict_action_one_turn.setdefault("general", {})
                    dict_action_one_turn["general"].setdefault("asking", {})
                    dict_action_one_turn["general"]["asking"].setdefault("none", "none")
            else:
                for slots_act in belief_state:
                    slots = slots_act["slots"]
                    domain_slot = slots[0][0].split("-")
                    domain = domain_slot[0]
                    slot = domain_slot[1]
                    value = slots[0][1]
                    action = slots_act["act"]

                    if domain not in dict_action_one_turn.keys():
                        dict_action_one_turn.setdefault(domain, dict())
                    if action not in dict_action_one_turn[domain].keys():
                        dict_action_one_turn[domain].setdefault(action, dict())

                    if domain in all_domains:
                        onto_mapping = self.map_ontology(domain, list_ontology)
                        for slot_digit, description_listslots in onto_mapping.items():
                            for description, listslots in description_listslots.items():
                                if slot in listslots:
                                    slot = slot_digit
                        dict_action_one_turn[domain][action].setdefault(slot, value)

            current_action = dict()
            for domain in dict_action_one_turn.keys():
                if domain in previous_state.keys() and "inform" in previous_state[domain].keys():
                    for slot,value in dict_action_one_turn[domain]["inform"].items():
                        if slot not in previous_state[domain]["inform"].keys() or value != previous_state[domain]["inform"][slot]:
                            if domain not in current_action.keys():
                                current_action.setdefault(domain, dict())
                            if "inform" not in current_action[domain].keys():
                                current_action[domain].setdefault("inform", dict())
                            current_action[domain]["inform"].setdefault(slot, value)
                else:
                    if domain not in current_action.keys():
                        current_action.setdefault(domain, dict())
                    if "inform" not in current_action[domain].keys() and "inform" in dict_action_one_turn[domain].keys():
                        current_action[domain].setdefault("inform", dict_action_one_turn[domain]["inform"])

                for action in dict_action_one_turn[domain].keys():
                    if action != "inform":
                        if domain not in current_action.keys():
                            current_action.setdefault(domain, dict())
                        current_action[domain][action] = dict_action_one_turn[domain][action]
            previous_state = dict_action_one_turn

            list_domain_action = []
            for domain, action_slot_value in current_action.items():
                list_asv = []
                domain_action = domain.upper() + ':['
                for action, slot_value in action_slot_value.items():
                    for slot, value in slot_value.items():
                        list_asv.append(action+'('+slot+'='+value+')')
                domain_action+= ' and '.join(p for p in list_asv) + ']'
                list_domain_action.append(domain_action)
            item['label'] = ' || '.join(dasv for dasv in list_domain_action)
            if "general" in item['label'].lower() or "[]" in item['label'].lower():
                item['label'] = ""

            list_current_domains = current_action.keys()
            list_gold_domain = self.get_list_gold_domain(all_domains, list_current_domains, list_ontology)
            item['ontology'] = ' || '.join(gold_domain for gold_domain in list_gold_domain).strip()

            list_sub_sample.append(item)
        return list_sub_sample

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

    def map_ontology(self, domain, ontologies, count=0):
        map_ontology_domain = {}
        for description, lists_slot in ontologies[domain.lower()].items():
            map_ontology_domain.setdefault("slot" + str(count), {description: lists_slot})
            count = count + 1
        return map_ontology_domain
        # {  "slot0":{"area to search for attractions": ["area"]},
        #    "slot1":{"name of the attraction": ["name"],
        #    "slot2":{"type of the attraction": ["type"]}}

    def get_ontology(self, domain_name, ontologies):
        onto_mapping = self.map_ontology(domain_name, ontologies)
        tmps = []
        for slotstr, description_listslots in onto_mapping.items():
            tmps.append(slotstr + "=" + list(description_listslots.keys())[0])

        value_onto = domain_name.upper() + ":(" + '; '.join(tmp for tmp in tmps) + ")"
        return value_onto, onto_mapping
        # value_onto = DOMAIN:(slot0=des0,slot1=des1,slot2=des2)

    def get_list_gold_domain(self, all_domains, list_current_domains, ontology):
        gold_domain = []
        tmp_domain = set()
        if len(list_current_domains)>0:
            domains = list_current_domains
        else:
            domains = all_domains
        for domain in domains:
            tmp_domain.add(domain.lower())
        for domain in tmp_domain:
            if domain in list_woz_domain:
                value_onto, _ = self.get_ontology(domain, ontology)
                gold_domain.append(value_onto)
        return gold_domain

if __name__ == '__main__':
    # TEST
    fusedchat_converter = MW24Converter(
        file_path=r'C:\ALL\GRADIENT\SERVER\gradient_server_test\data\raw\MW24',
        save_path=r"C:\ALL\GRADIENT\SERVER\gradient_server_test\data\interim\GradSearch\GradSearch_v1\MW24").__call__(
        instruction_path=r"C:\ALL\GRADIENT\SERVER\gradient_server_test\data\instructions\instruct_GradSearch_old.txt",
        ontolopy_path=r"C:\ALL\GRADIENT\SERVER\gradient_server_test\data\schema_guided.json")







