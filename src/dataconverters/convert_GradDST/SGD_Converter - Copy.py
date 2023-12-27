import sys
import json
import random
import glob
import os
from typing import List, Dict, Union, Optional
from src.dataconverters.converter import DialConverter

sys.path.append('/content/drive/MyDrive/Colab Notebooks/baseline_v1/gradients.baselinev1.dialogstate')
list_sgd_domain = ['buses_1', 'buses_2', 'buses_3', 'calendar_1', 'events_1', 'events_2', 'events_3', 'flights_3',
                   'flights_1', 'flights_2', 'flights_4', 'homes_1', 'homes_2', 'hotels_1', 'hotels_2', 'banks_2',
                   'hotels_3', 'hotels_4', 'media_1', 'media_3', 'messaging_1', 'movies_1', 'movies_3', 'movies_2',
                   'music_1', 'music_2', 'music_3', 'payment_1', 'rentalcars_1', 'rentalcars_2', 'rentalcars_3',
                   'restaurants_1', 'restaurants_2', 'ridesharing_1', 'ridesharing_2', 'services_1', 'banks_1',
                   'services_2', 'services_3', 'services_4', 'trains_1', 'travel_1', 'weather_1', 'alarm_1', 'media_2']

list_user_action = ['inform', 'request', 'inform_intent', 'negate_intent', 'affirm_intent', 'affirm', 'negate', 'select', 'thank_you', 'goodbye', 'greet', 'general_asking', 'request_alts']

class SGDConverter(DialConverter):
    def __init__(self,
                 file_path: str,
                 save_path: str,
                 tag_user: str = 'USER',
                 tag_system: str = 'AGENT',
                 window_context: int = 5  # slide window
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

    def process(self, instruction_path, ontolopy_path) -> None:
        # Separate complete dialogues to sub dialogues
        list_data_path = glob.glob(os.path.join(self.file_path, '*.json'))
        list_instruction = self.load_instruction(instruction_path)
        list_ontology = self.load_ontology(ontolopy_path)

        for data_path in list_data_path:
            filename = os.path.basename(data_path)
            dataset = self.load_datapath(data_path)

            list_all_sample = []
            cw = self.window_context - 1  # slide window
            # Analyze all dialogues
            for dialogue in dataset:
                id = dialogue["dialogue_id"]
                # process dialogue into sub-dialogues
                list_sub_dialogue = []
                for i in range(len(dialogue['turns'])):
                    if i % 2 == 0:
                        list_sub_dialogue.append(dialogue['turns'][max(0, i - cw):max(0, i + 1)])
                # process raw list_sub_dialogue to interim list_sub_sample
                list_sub_sample = self.get_list_sub_sample(id, dialogue["services"], list_sub_dialogue, list_ontology,list_instruction)
                list_all_sample.extend(list_sub_sample)
            self.save_datapath(list_all_sample, filename)

    def get_list_sub_sample(self, id, all_domains, list_sub_dialogue, list_ontology, list_instruction):
        list_sub_sample = []
        current_values = set()
        previous_state = dict()
        for id_turn, sub_dialogue in enumerate(list_sub_dialogue):
            item = dict()
            dict_action_one_turn = dict()

            # get context, current_user, instruction, list_user_action and ontology
            list_turn = []
            for turn in sub_dialogue:
                speaker = self.tag_user if turn['speaker'] == 'USER' else self.tag_system
                list_turn.append(speaker + ": " + turn['utterance'])

            item['instruction'] = self.get_instruction(list_instruction).strip()
            item['list_user_action'] = ', '.join(action.lower() for action in list_user_action)
            item['ontology'] = ''
            item['id_dialogue'] = id
            item['id_turn'] = id_turn * 2 + 1
            item['context'] = ' '.join([list_turn[i].strip() for i in range(len(list_turn)-1)]).strip()
            item['current_query'] = list_turn[-1].strip()
            if item['context'] != "":
                item['current_query'] = item['current_query'][6:]


            # get type, current_state and current_action
            frames = sub_dialogue[-1]['frames']
            if len(sub_dialogue)>1:
                system_frames = sub_dialogue[-2]['frames']

            for frame in frames:
                domain = frame["service"].lower()
                actions = frame["actions"]
                state = frame['state']
                slot_values = state["slot_values"]
                onto_mapping = self.map_ontology(domain, list_ontology)

                for action in actions:
                    act = action["act"].lower().strip()
                    slot = action["slot"].strip().lower()
                    for slotstr, description_listslots in onto_mapping.items():
                        for description, listslots in description_listslots.items():
                            if slot in listslots:
                                slot = slotstr
                    if "slot" not in slot and slot != "intent" and slot !="":
                        print(act, slot, action["values"])
                    if act == "request":
                        value = "?"
                    else:
                        if len(action["values"])>0:
                            value = action["values"][0]
                        else:
                            value = "none"
                    if value not in ["?", "none"]:
                        current_values.add(value)
                    if domain not in dict_action_one_turn.keys():
                        dict_action_one_turn.setdefault(domain, dict())
                    if act not in dict_action_one_turn[domain].keys():
                        dict_action_one_turn[domain].setdefault(act, dict())
                    if slot not in dict_action_one_turn[domain][act].keys():
                        dict_action_one_turn[domain][act].setdefault(slot, value)
                    if value != dict_action_one_turn[domain][act][slot]:
                        dict_action_one_turn[domain][act][slot] = value

            current_action = dict()
            for domain in dict_action_one_turn.keys():
                if domain in previous_state.keys() and "inform" in previous_state[domain].keys():
                    if "inform" in dict_action_one_turn[domain].keys():
                        for slot, value in dict_action_one_turn[domain]["inform"].items():
                            if slot not in previous_state[domain]["inform"].keys() or value != \
                                    previous_state[domain]["inform"][slot]:
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
                        if slot in ["intent"]:
                            list_asv.append(action + '(' + value + ')')
                        else:
                            if action in ["select", "affirm", "goodbye", "thank_you", "negate", "request_alts",
                                          "negate_intent", "affirm_intent"]:
                                list_asv.append(action)
                            elif action in ["request"]:
                                list_asv.append(action + '(' + slot +')')
                            else:
                                list_asv.append(action + '(' + slot + '=' + value + ')')
                domain_action += ' and '.join(p for p in list_asv) + ']'
                list_domain_action.append(domain_action)
            item['label'] = ' || '.join(dasv for dasv in list_domain_action)
            if "general" in item['label'].lower() or "[]" in item['label'].lower():
                item['label'] = ""
            list_current_domains = current_action.keys()
            list_gold_domain = self.get_list_gold_domain(all_domains, list_current_domains, list_ontology)
            item['ontology'] = '||'.join(gold_domain for gold_domain in list_gold_domain).strip()
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
        if len(list_current_domains) > 0:
            domains = list_current_domains
        else:
            domains = all_domains
        for domain in domains:
            tmp_domain.add(domain.lower())
        for domain in tmp_domain:
            if domain in list_sgd_domain:
                value_onto, _ = self.get_ontology(domain, ontology)
                gold_domain.append(value_onto)
        return gold_domain

if __name__ == '__main__':
    # TEST
    sgd_converter = SGDConverter(
        file_path=r'C:\ALL\GRADIENT\SERVER\gradient_server_test\data\raw\SGD',
        save_path=r'C:\ALL\GRADIENT\SERVER\gradient_server_test\data\interim\GradSearch\GradSearch_v1\SGD').__call__(
        instruction_path=r"C:\ALL\GRADIENT\SERVER\gradient_server_test\data\instructions\instruct_GradSearch_old.txt",
        ontolopy_path=r"C:\ALL\GRADIENT\SERVER\gradient_server_test\data\schema_guided.json")












