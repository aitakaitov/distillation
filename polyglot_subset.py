from datasets import load_dataset, concatenate_datasets

languages = ['en', 'de', 'fr', 'pl', 'sk', 'el', 'vi', 'hu', 'es', 'ar', 'zh', 'ru', 'cs']

datasets = []

for lang in languages:
    datasets.append(load_dataset('polyglot_ner', split='train', name=lang))

merged_dataset = datasets[0]

for dataset in datasets[1:]:
    merged_dataset = concatenate_datasets([merged_dataset, dataset])

merged_dataset.save_to_disk('polyglot_merged')