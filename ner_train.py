import numpy as np
import torch.backends.cuda
from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import argparse
import wandb
import time

#torch.backends.cuda.matmul.allow_tf32 = True

############################### FORCE CPU
# torch.cuda.is_available = lambda: False


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required=True, default=2, type=int)
parser.add_argument('--model', required=True, default='xlm-roberta-large', type=str)
parser.add_argument('--lr', required=True, default=1e-5, type=float)
parser.add_argument('--batch_size', required=False, default=1, type=int)
parser.add_argument('--test_split_size', required=False, default=0.05, type=float)
parser.add_argument('--cache_file', required=False, default=None, type=str)

args = parser.parse_args()

h = str(time.time_ns())
wandb.init(project='rembert', entity='aitakaitov', tags=[h], config={
    'lr': args.lr,
    'batch_size': args.batch_size,
    'model': args.model
})

label_list = ['O', 'I-LOC', 'B-LOC', 'I-ORG', 'B-ORG', 'I-PER', 'B-PER']
label_encoding_dict = {'O': 0, 'I-LOC': 1, 'B-LOC': 2, 'I-ORG': 3, 'B-ORG': 4, 'I-PER': 5, 'B-PER': 6}

task = "ner"
model_checkpoint = args.model
batch_size = args.batch_size

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_and_align_labels(batch):
    tokens = batch['words']
    ner_tags = batch['ner']
    tokenized_inputs = tokenizer(list(tokens), truncation=True, is_split_into_words=True, max_length=128)
    word_ids = tokenized_inputs.word_ids()

    current_word = None
    new_labels = []
    for i, seq in enumerate(tokenized_inputs.encodings):
        new_seq_labels = []
        word_ids = seq.word_ids
        seq_ner_tags = ner_tags[i]
        for word_id in word_ids:
            if word_id != current_word:
                current_word = word_id
                label = -100 if word_id is None else label_encoding_dict[seq_ner_tags[word_id]]
                new_seq_labels.append(label)
            elif word_id is None:
                # special token = -100
                new_seq_labels.append(-100)
            else:
                new_seq_labels.append(label_encoding_dict[seq_ner_tags[word_id]])

        new_labels.append(new_seq_labels)

    tokenized_inputs['labels'] = new_labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    labels = np.squeeze(labels)

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    wandb.log({"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]})

    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}


if __name__ == '__main__':
    start_time = time.time_ns()

    dataset = load_dataset('json', data_files='polyglot_processed.json')['train'] \
                .map(tokenize_and_align_labels, batched=True, num_proc=None, cache_file_name=args.cache_file)
    dataset = dataset.shuffle(seed=42)

    split_dataset = dataset.train_test_split(test_size=args.test_split_size, shuffle=True, seed=42)
    train_dataset, test_dataset = split_dataset['train'], split_dataset['test']

    print(f'Dataset mapping complete after {(time.time_ns() - start_time) / 1_000_000_000 / 60} mins')

    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

    args = TrainingArguments(
        f'{args.model}_{args.lr}',
        evaluation_strategy='steps',
        eval_steps=45500,               # should be 10 times per epoch
        learning_rate=args.lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=args.epochs,
        weight_decay=1e-5,
        fp16=True,
        #tf32=True,
        save_strategy='epoch',
        save_steps=45500,
        group_by_length=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True, pad_to_multiple_of=8)
    metric = load_metric("seqeval")

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()

    trainer.save_model(h)


