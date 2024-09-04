import functools
import json
import os

import datasets
import pandas as pd
from datasets import Dataset

# import _settings

DATA_FOLDER = "/disk1/chenchao/Code/UQ-NLG/data/datasets"

def _save_dataset():
    # https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/parse_coqa.py
    save_path = f'{DATA_FOLDER}/SQuAD'
    if not os.path.exists(save_path):
        # https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json
        with open('{}/dev-v2.0.json'.format(DATA_FOLDER), 'r') as infile:
            data = json.load(infile)['data']

        dataset = {}

        dataset['story'] = []
        dataset['question'] = []
        dataset['answer'] = []
        dataset['additional_answers'] = []
        dataset['id'] = []

        for _data in data:
            paragraphs = _data["paragraphs"]
            for sample_id, sample in enumerate(paragraphs):
                print(sample)
                story = sample['context']
                questions = sample['qas']
                # answers = sample['answers']
                # additional_answers = sample['additional_answers']
                for question_index, question in enumerate(questions):
                    if question["is_impossible"]:
                        continue
                    dataset['story'].append(story)
                    dataset['question'].append(question['question'])
                    dataset['answer'].append({
                        'text': question["answers"][0]['text'],
                        'answer_start': question["answers"][0]['answer_start']
                    })
                    dataset['id'].append(question['id'])
                    additional_answers_list = []
                    for i in range(len(question["answers"])):
                        additional_answers_list.append(question["answers"][i]['text'])
                    dataset['additional_answers'].append(additional_answers_list)

        dataset_df = pd.DataFrame.from_dict(dataset)

        dataset = Dataset.from_pandas(dataset_df)

        dataset.save_to_disk(save_path)
    return save_path

@functools.lru_cache(1)
def read_all_contexts():
    dataset = datasets.load_from_disk(_save_dataset())
    return {_['id']: _['story'] for _ in dataset}



def get_dataset(tokenizer, split='validation'):
    # from https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/parse_coqa.py
    dataset = datasets.load_from_disk(_save_dataset())
    id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))

    def encode_coqa(example):
        example['answer'] = example['answer']['text']
        example['prompt'] = prompt = example['story'] + ' Q: ' + example['question'] + ' A:'
        return tokenizer(prompt, truncation=False, padding=False)
    dataset = dataset.map(encode_coqa, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
    return dataset



def _generate_config(tokenizer):

    if tokenizer.__class__.__name__ == 'LlamaTokenizer':
        eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']] + [29889]  # seems to be '.' as well
        #eos_token_id = [tokenizer(_)['input_ids'] for _ in ['\n', ',', '.']]
    elif tokenizer.__class__.__name__ == 'GPT2Tokenizer':
        eos_token_id = [tokenizer.encode(_)[1] for _ in ['.', '\n']]
    elif tokenizer.__class__.__name__ == "PreTrainedTokenizerFast":
        eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']] + [29889] 
    else:
        raise NotImplementedError
    eos_token_id += [tokenizer.eos_token_id]
    question_framing_ids = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
    # Follows Kuhn et al 2023 as Llama does not have CoQA
    question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in question_framing_ids]
    return dict(eos_token_id=eos_token_id, bad_words_ids=question_framing_ids)

if __name__ == '__main__':
    import models
    dataset = get_dataset(models.load_tokenizer())