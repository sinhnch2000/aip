import sys
import json
import random
import glob
import os
from typing import List, Dict, Union, Optional
from src.dataconverters.converter import DialConverter

sys.path.append('/content/drive/MyDrive/Colab Notebooks/baseline_v1/gradients.baselinev1.dialogstate')
list_woz_domain = ['taxi', 'police', 'hospital', 'hotel', 'attraction', 'train', 'restaurant']
list_user_action = ['INFORM', 'REQUEST','NEGATE','THANK', 'BYE','GREET','ASKING','REQUEST_ALTS']

class FushedChatConverter(DialConverter):
    def __init__(self,
                 file_path: str,
                 save_path: str,
                 tag_user: str = 'USER',
                 tag_system: str = 'SYSTEM',
                 ) -> None:
        """
        Args:
            save_path: path to save the processed dataset
        """
        super().__init__(file_path, save_path, tag_user, tag_system)

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
            # Analyze all dialogues
            for id, dialogue in dataset.items():

                # get summarize dialogue
                list_gold_domain = self.get_list_gold_domain(dialogue, list_ontology)
                # process dialogue into sub-dialogues
                list_sub_dialogue = []
                for i in range(len(dialogue['log'])):
                    if i % 2 == 0:
                        list_sub_dialogue.append(dialogue['log'][0:i+1])
                # process raw list_sub_dialogue to interim list_sub_sample
                list_sub_sample = self.get_list_sub_sample(id, list_sub_dialogue, list_gold_domain, list_ontology, list_instruction)
                list_all_sample.extend(list_sub_sample)
            self.save_datapath(list_all_sample, filename)

    def get_list_sub_sample(self, id, list_sub_dialogue, list_gold_domain, list_ontology, list_instruction):
        list_sub_sample = []
        dict_state_one_dialogue = dict()
        for id_turn, sub_dialogue in enumerate(list_sub_dialogue):
            item = dict()
            list_current_state = list()
            list_current_action = list()
            set_type = set()
            dict_action_one_turn = dict()

            # get context, current_user, instruction, list_user_action and ontology
            list_turn = []
            for turn in sub_dialogue:
                speaker = self.tag_user if turn['speaker'] == 'USER' else self.tag_system
                list_turn.append(speaker + ": " + turn['text'])

            item['instruction'] = self.get_instruction(list_instruction).strip()
            item['list_user_action'] = '[' + ', '.join(action.lower() for action in list_user_action) + ']'
            item['ontology'] = ' & '.join(gold_domain for gold_domain in list_gold_domain).strip()
            item['history'] = ' '.join([list_turn[i].strip() for i in range(len(list_turn)-1)]).strip()
            item['current'] = list_turn[-1].strip()
            item['id_dialogue'] = id
            item['id_turn'] = id_turn*2 + 1

            # get type, current_state and current_action
            dialog_act = sub_dialogue[-1]['dialog_act']

            if len(dialog_act) == 0: # ODD || "dialog_act": {},
                if "asking" not in dict_action_one_turn.keys():
                    dict_action_one_turn.setdefault("asking", {})
                dict_action_one_turn["asking"].setdefault("general", {})
                dict_action_one_turn["asking"]["general"].setdefault("none", "none")
                set_type.add("odd")
            else:
                for domain_action, frame in dialog_act.items():
                    # domain_action: domain | action
                    # frame: [[slot, value], [slot, value]]
                    domain_action = domain_action.split('-')
                    domain = domain_action[0].lower()

                    if domain == 'chitchat': # ODD || "dialog_act": {"chitchat": []}
                        if "asking" not in dict_action_one_turn.keys():
                            dict_action_one_turn.setdefault("asking",dict())
                        dict_action_one_turn["asking"].setdefault("general",dict())
                        dict_action_one_turn["asking"]["general"].setdefault("none", "none")
                        set_type.add("odd")
                    else:
                        action = domain_action[1].lower()
                        if action not in dict_action_one_turn.keys():
                            dict_action_one_turn.setdefault(action, dict())
                        dict_action_one_turn[action].setdefault(domain, dict())
                        if domain in list_woz_domain: # TOD
                            if domain in item['ontology']:
                                if len(frame) == 0:
                                    dict_action_one_turn[action][domain].setdefault("none", "none")
                                else:
                                    for slot_value in frame:
                                        slot = slot_value[0].lower()
                                        onto_mapping = self.map_ontology(domain, list_ontology)
                                        for slot_digit, description_listslots in onto_mapping.items():
                                            for description, listslots in description_listslots.items():
                                                if slot in listslots:
                                                    slot = slot_digit
                                        value = slot_value[1].lower()

                                        dict_action_one_turn[action][domain].setdefault(slot, value)
                                set_type.add("tod")

                        else: # ODD (general-thank | general-bye | general-greet) just a state not a domain
                            dict_action_one_turn[action][domain].setdefault("none", "none")
                            set_type.add("odd")

            metadata = sub_dialogue[-1]['metadata']
            for domain, state in metadata.items():
                for dict_slot_value in state.values():
                    for slot, value in dict_slot_value.items():
                        if value not in ["", "not mentioned"] and slot != "booked":
                            onto_mapping = self.map_ontology(domain, list_ontology)
                            for slot_digit, description_listslots in onto_mapping.items():
                                for description, listslots in description_listslots.items():
                                    if slot.lower() in listslots:
                                        slot = slot_digit
                            if domain == "bus" and "TRAIN" in item['ontology']:
                                domain = "train"
                            if domain != "bus" and domain in item['ontology']:
                                if domain not in dict_state_one_dialogue.keys():
                                    if "inform" not in dict_action_one_turn.keys():
                                        dict_action_one_turn.setdefault("inform", dict())
                                    if domain not in dict_action_one_turn["inform"].keys():
                                        dict_action_one_turn["inform"].setdefault(domain, dict())
                                    if slot not in dict_action_one_turn["inform"][domain].keys():
                                        dict_action_one_turn["inform"][domain].setdefault(slot, value)
                                else:
                                    if slot not in dict_state_one_dialogue[domain].keys():
                                        if "inform" not in dict_action_one_turn.keys():
                                            dict_action_one_turn.setdefault("inform", dict())
                                        if domain not in dict_action_one_turn["inform"].keys():
                                            dict_action_one_turn["inform"].setdefault(domain, dict())
                                        if slot not in dict_action_one_turn["inform"][domain].keys():
                                            dict_action_one_turn["inform"][domain].setdefault(slot, value)

            if "inform" in dict_action_one_turn.keys():
                for domain, dict_slot_value in dict_action_one_turn["inform"].items():
                    if domain in list_woz_domain:
                        if domain not in dict_state_one_dialogue.keys():
                            dict_state_one_dialogue.setdefault(domain, dict())
                        for slot, value in dict_slot_value.items():
                            if slot != "none" and value != "none":
                                if slot not in dict_state_one_dialogue[domain].keys():
                                    dict_state_one_dialogue[domain].setdefault(slot, "")
                                dict_state_one_dialogue[domain][slot] = value

            for domain, dict_slot_value in dict_state_one_dialogue.items():
                for slot, value in dict_slot_value.items():
                    list_current_state.append(domain + '-' + slot + '-' + value)

            for action, dict_domain_slot_value in dict_action_one_turn.items():
                for domain, dict_slot_value in dict_domain_slot_value.items():
                    for slot, value in dict_slot_value.items():
                        list_current_action.append(action + '>' + domain + '-' + slot + '-' + value)

            final_type = "tod" if "tod" in set_type else "odd"

            if len(list_current_action)>0:
                final_current_action = ' || '.join(current_action for current_action in list_current_action).lower().strip()
            else:
                final_current_action = "asking>general-none-none"

            if len(list_current_state) > 0:
                final_current_state = ' || '.join(current_state for current_state in list_current_state).lower().strip()
            else:
                final_current_state = "nothing"

            item['label'] = "(type) " + final_type + " (current action) " + final_current_action.replace("dont care", "dontcare") + " (current state) " + final_current_state.replace("dont care", "dontcare")
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

        value_onto = domain_name + ":[" + ' | '.join(tmp for tmp in tmps) + "]"
        return value_onto, onto_mapping
        # value_onto = DOMAIN:(slot0=des0,slot1=des1,slot2=des2)

    def get_list_gold_domain(self, dialogue, ontology):
        gold_domain = []
        tmp_domain = set()
        goal = dialogue["goal"]
        for domain, all_info in goal.items():
            if domain not in ["message", "topic"] and all_info != {}:
                if "info" in all_info.keys():
                    tmp_domain.add(domain.lower())
        for domain in tmp_domain:
            if domain in list_woz_domain:
                value_onto,_ = self.get_ontology(domain, ontology)
                gold_domain.append(value_onto)
        return gold_domain


if __name__ == '__main__':
    # TEST
    fusedchat_converter = FushedChatConverter(
        file_path=r'C:\ALL\OJT\SERVER\gradient_server_test\data\data raw\FUSEDCHAT',
        save_path=r'C:\ALL\OJT\SERVER\gradient_server_test\data\data interim\GradSearch\GradSearch_v2\FUSEDCHAT').__call__(
        instruction_path=r"C:\ALL\OJT\SERVER\gradient_server_test\data\instructions\instruct_GradSearch.txt",
        ontolopy_path=r"C:\ALL\OJT\SERVER\gradient_server_test\data\schema guided\schema.json")







