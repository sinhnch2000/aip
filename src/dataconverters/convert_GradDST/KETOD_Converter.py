import sys
import json
import random
import glob
import os
from typing import List, Dict, Union, Optional
from src.features.converter import DialConverter

sys.path.append('/content/drive/MyDrive/Colab Notebooks/baseline_v1/gradients.baselinev1.dialogstate')
list_woz_domain = ['taxi', 'police', 'hospital', 'hotel', 'attraction', 'train', 'restaurant']
list_user_action = ['INFORM', 'REQUEST','INFORM_INTENT','NEGATE_INTENT',
                    'AFFIRM_INTENT', 'AFFIRM','NEGATE','SELECT','THANK',
                    'BYE','GREET','ASKING','REQUEST_ALTS']


class KetodConverter(DialConverter):
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

    def process(self, instruction_path, ontolopy_path) -> None:
        # Separate complete dialogues to sub dialogues
        list_data_path = glob.glob(os.path.join(self.file_path, '*.json'))
        list_instruction = self.load_instruction(instruction_path)
        list_ontology = self.load_ontology(ontolopy_path)

        for data_path in list_data_path:
            filename = os.path.basename(data_path)
            dataset = self.load_datapath(data_path)

            list_all_sample = []
            # Analyze all dialogues
            for dialogue in dataset.values():
                # get summarize dialogue
                list_gold_domain = self.get_list_gold_domain(dialogue, list_ontology)
                # process dialogue into sub-dialogues
                list_sub_dialogue = []
                for i in range(len(dialogue['log'])):
                    if i % 2 == 0:
                        list_sub_dialogue.append(dialogue['log'][0:i + 1])
                # process raw list_sub_dialogue to interim list_sub_sample
                list_sub_sample = self.get_list_sub_sample(list_sub_dialogue, list_gold_domain, list_ontology,
                                                           list_instruction)
                list_all_sample.extend(list_sub_sample)
            self.save_datapath(list_all_sample, filename)

    def get_sub_dialouge(self, dataset):
        """
        This function is to get utterance (<=5 utterance)
        :return: the list of list, each list contain utterances for each Input
                EX: [[utterance1, utterance2, utterance3, ...],
                    [utterance4, utterance5, utterance6, ...]]
        """
        list_utter = []
        cw = self.window_context - 1  # slide window
        for dialogue in dataset:
            list_turns = dialogue['turns']
            for idx in range(len(list_turns)):
                if idx % 2 == 0:  # current_utter is user
                    child_dialogue = list_turns[max(0, idx - cw):max(0, idx + 1)]
                    list_utter.append(child_dialogue)
        return list_utter

    def map_ontology(self, ontologies, domain, count=0):
        map_ontology_domain = {}
        for slot in ontologies[domain].keys():
            map_ontology_domain.setdefault(slot, "slot" + str(count))
            count = count + 1
        return map_ontology_domain

    def get_ontology(self, dialogue, ontologies):
        results, onto_mapping_ls = [], []
        for idx, frame in enumerate(dialogue[-1]["frames"]):
            # get domain (domain == service)
            domain = frame["service"].strip().lower()
            onto_mapping = self.map_ontology(ontologies, domain)
            tmps = [onto_mapping[key]  + "=" + value 
                for key, value in ontologies[domain].items()]
            value_onto = domain.upper() + ":(" + '; '.join(tmp for tmp in tmps) + ")"

            results.append(value_onto)
            onto_mapping_ls.append(onto_mapping)

        return results, onto_mapping_ls
    
    def define_ontology(self, ontolopy_path: Optional[str] = None) -> Union[Dict, List[Dict]]:
        with open(ontolopy_path) as f:
            ontologies = json.load(f)
        return ontologies

    def define_instruct(self, instruct_path) -> List[str]:
        with open(instruct_path) as f:
            instructions = f.readlines()
        return instructions
    
    def save_datapath(self, data_processed: List[Dict], filename: str):
        with open(os.path.join(self.save_path, filename), 'w') as f:
            json.dump(data_processed, f, indent=4)

    def load_datapath(self, data_path) -> List[Dict]:
        with open(data_path, 'r+') as f:
            dataset = json.load(f)
        return dataset


if __name__ == '__main__':
    converter = KetodConverter(file_path='/content/raw/KETOD',
                                save_path='/content/interim/GradSearch/KETOD',
                                window_context=5)
    converter.__call__(instruct_path='./data/instruct_GradSearch.txt',
                        ontolopy_path='./data/schema_guided.json')












