import re

from typing import Optional, List, Union, Set
from datasets import DatasetDict, load_dataset, concatenate_datasets

class StateProcessing:
    def __init__(self,
                 tokenizer: str,
                 train_file: Optional[Union[str, List[str]]],
                 val_file: Optional[Union[str, List[str]]],
                 test_file: Optional[Union[str, List[str]]]=None,
                 batch_size: int = 8,
                 max_train_samples: Optional[int] = None,
                 max_eval_samples: Optional[int] = None,
                 max_predict_samples: Optional[int] = None
                 ) -> None:

        self.tokenizer = tokenizer

        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size

        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.max_predict_samples = max_predict_samples

    def __call__(self, *args, **kwargs):
        dataset = {}

        if self.train_file is not None:
            print('\nLoading train datasets' + '.' * 10)
            train_dataset = self.load_data('train', self.train_file)
            if self.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.max_train_samples))
            dataset['train'] = self.process_fn(train_dataset)

        if self.val_file is not None:
            print('\nLoading validation datasets' + '.' * 10)
            eval_dataset = self.load_data('val', self.val_file)
            if self.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(self.max_eval_samples))
            dataset['eval'] = self.process_fn(eval_dataset)

        if self.test_file is not None:
            print('\nLoading test datasets' + '.' * 10)
            test_dataset = self.load_data('test', self.test_file)
            if self.max_predict_samples is not None:
                test_dataset = test_dataset.select(range(self.max_predict_samples))
            dataset['test'] = self.process_fn(test_dataset)

        return dataset

    def load_data(self, key: str, data_file: List[str]) -> DatasetDict:
        """
        Loads a dataset from a file on disk and returns it as a dictionary of Dataset objects.

        Args:
            key (str): The key to assign to the loaded dataset in the returned dictionary of Dataset objects.
            data_file (Union[str, List[str]]): The path or paths to the data file(s) to load. If multiple is True,
                        data_file should be a list of file paths. Otherwise, it should be a single file path.
            mutiple (bool): A flag that indicates whether the data_file argument is a list of multiple file paths.

        Returns:
            A dictionary of Dataset objects that represents the loaded dataset. If mutiple is True, the function
            concatenates the datasets from the multiple files before returning them. Otherwise, it returns a single
            dataset loaded from the data_file path.
        """
        dataset = load_dataset('json', data_files=f'{data_file}/*.json',split='train')
        dataset.shuffle(42)
        return dataset

    def tokenizer_fn(self, batch_samples):
        """
        A collate function that tokenizes the inputs and targets, and applies dynamic padding and truncation
        based on the maximum length in the batch.

        Args:
            batch (list): A list of examples, where each example is a dictionary with a text column and a target column.

        Returns:
            dict: A dictionary with the input IDs, attention masks, and target IDs with attention masks
            where tokens are padded, and the target IDs are masked to exclude padded values.
        """

        def mapping_sample(examples):
            inputs, targets = [], []
            for idx in range(len(examples['instruction'])):
                item = examples['instruction'][idx] \
                    .replace('{list_user_action}', examples['list_user_action'][idx].strip()) \
                    .replace('{history}', examples['history'][idx].strip()) \
                    .replace('{current}', examples['current'][idx].strip()) \
                    .replace('{ontology}', examples['ontology'][idx].strip()) \

                inputs.append(re.sub('\s+', ' ', item))
                targets.append(re.sub('\s+', ' ', examples['label'][idx].strip()))
            return inputs, targets

        inputs, targets = mapping_sample(batch_samples)

        model_inputs = self.tokenizer(inputs, padding='longest')
        tgt_tokens = self.tokenizer(targets, padding='longest')

        tgt_tokens["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in tgt_tokens["input_ids"]
            ]

        model_inputs['labels'] = tgt_tokens["input_ids"]

        return model_inputs

    def process_fn(self, dataset):
        dataset = dataset.map(
            lambda example: self.tokenizer_fn(example),
            batched=True, batch_size=self.batch_size,
            num_proc= 4,
            load_from_cache_file=False,
            remove_columns=['list_user_action', 'history', 'current', 'ontology','instruction', 'id_turn', 'id_dialogue', 'label'],
            desc= 'Tokenizer processing'
        )

        return dataset