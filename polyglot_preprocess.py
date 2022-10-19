import json

from datasets import load_from_disk

from tqdm import tqdm

dataset = load_from_disk("polyglot_merged")
languages = ['en', 'de', 'fr', 'pl', 'sk', 'el', 'vi', 'hu', 'es', 'ar', 'zh', 'ru', 'cs']

with open("polyglot_processed.json", "w", encoding='utf-8') as f:
    for sample in tqdm(
            dataset.filter(lambda sample: sample['lang'] in languages).map(lambda sample:
                                                                        {
                                                                            "id": sample['id'],
                                                                            "lang": sample['lang'],
                                                                            "words": sample['words'],
                                                                            "ner": sample['ner']
                                                                        },
                                                                        new_fingerprint="po mappingu"
                                                                        )
    ):
        prev_entity = "O"
        new_ner = []
        for i, (word, entity) in enumerate(zip(sample['words'], sample['ner'])):
            if entity == "O":
                mod_entity = entity
            else:
                if entity == prev_entity:
                    mod_entity = "I-" + entity
                else:
                    mod_entity = "B-" + entity

            new_ner.append(mod_entity)
            prev_entity = entity

        print(json.dumps(
            {
                'id': sample['id'],
                'words': sample['words'],
                'lang': sample['lang'],
                'ner': new_ner
            }
        ), file=f)
