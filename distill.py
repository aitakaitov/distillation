import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer
import argparse
from datasets import load_metric, load_dataset
import numpy as np
import wandb
import time

DATASET_SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument('--lr', required=True, type=float)
parser.add_argument('--batch_size', required=False, type=int, default=1)
parser.add_argument('--epochs', required=True, type=int)
parser.add_argument('--teacher_model', required=True, type=str)
parser.add_argument('--student_model', required=True, type=str)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--temperature', required=False, type=float, default=1.0)
parser.add_argument('--weight_decay', required=False, type=float, default=1e-5)
parser.add_argument('--loss', required=False, type=str, default='ce_kl', help='ce_kl')
parser.add_argument('--distill_loss_weight', required=False, type=float, default=0.7,
                    help='Weight of KLDiv loss when loss == ce_kl. CE loss weight = 1 - val')
parser.add_argument('--test_split_size', required=False, type=float, default=0.05)

args = parser.parse_args()

h = str(time.time_ns())

label_list = ['O', 'I-LOC', 'B-LOC', 'I-ORG', 'B-ORG', 'I-PER', 'B-PER']
label_encoding_dict = {'O': 0, 'I-LOC': 1, 'B-LOC': 2, 'I-ORG': 3, 'B-ORG': 4, 'I-PER': 5, 'B-PER': 6}


def load_model(model_path, num_labels):
    return AutoModelForTokenClassification.from_pretrained(model_path, num_labels=num_labels), \
           AutoTokenizer.from_pretrained(model_path)


class Distiller(torch.nn.Module):
    def __init__(self, student_model, teacher_model, temperature, loss_str):
        super(Distiller, self).__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.loss_str = loss_str

        self.kldivloss = KLDivLoss(reduction='batchmean')
        self.ce = CrossEntropyLoss()

        student_model.train()
        teacher_model.eval()

    def loss(self, student_logits, teacher_logits, student_label):
        if self.loss_str == 'ce_kl':
            return self.loss_ce_kl(student_logits, teacher_logits, student_label)
        else:
            raise RuntimeError(f'Loss {args.loss} is not valid.')

    def get_logits(self, input_ids, attention_mask, from_teacher=False):
        if from_teacher:
            teacher_output = self.teacher_model(input_ids, attention_mask)
            return teacher_output.logits
        else:
            student_output = self.student_model(input_ids, attention_mask)
            return student_output.logits

    def forward(self, teacher_input_ids, teacher_attention_mask, student_input_ids, student_attention_mask,
                student_labels, test=False):
        if not test:
            student_logits = self.get_logits(student_input_ids, student_attention_mask, False)
            teacher_logits = self.get_logits(teacher_input_ids, teacher_attention_mask, True)
            return torch.nn.functional.softmax(student_logits, dim=2), self.loss(teacher_logits, student_logits,
                                                                                 student_labels)
        else:
            student_logits = self.get_logits(student_input_ids, student_attention_mask, False)
            return torch.nn.functional.softmax(student_logits, dim=2)

    def loss_ce_kl(self, teacher_logits, student_logits, labels):
        student_logits, teacher_logits = (student_logits / self.temperature).softmax(1), (
                teacher_logits / self.temperature).softmax(1)

        loss = self.ce(student_logits.permute(0, 2, 1), labels) * (1 - args.distill_loss_weight)
        loss = loss + self.kldivloss()(student_logits, teacher_logits) * args.distill_loss_weight
        return loss


class NERDataset:
    @staticmethod
    def tokenize_and_align_labels(tokens, ner_tags, tokenizer, student=False):
        tokenized_inputs = tokenizer(list(tokens), truncation=True, is_split_into_words=True, padding='max_length',
                                     max_length=STUDENT_MAX_LENGTH if student else TEACHER_MAX_LENGTH)
        word_ids = tokenized_inputs.word_ids()

        current_word = None
        new_labels = []

        for word_id in word_ids:
            if word_id != current_word:
                current_word = word_id
                label = -100 if word_id is None else label_encoding_dict[ner_tags[word_id]]
                new_labels.append(label)
            elif word_id is None:
                # special token = -100
                new_labels.append(-100)
            else:
                new_labels.append(label_encoding_dict[ner_tags[word_id]])

        tokenized_inputs['label'] = new_labels
        return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    labels = np.squeeze(labels)

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                        zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                   zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    wandb.log(
        {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"],
         "accuracy": results["overall_accuracy"]})

    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]}


def create_batch(samples):
    return torch.LongTensor(samples)


def preprocess_samples(batch):
    teacher_input_ids_batch = []
    student_input_ids_batch = []
    teacher_att_mask_batch = []
    student_att_mask_batch = []
    student_labels_batch = []

    for sample in batch:
        teacher_input_aligned = NERDataset.tokenize_and_align_labels(sample['words'], sample['ner'], teacher_tokenizer)
        student_input_aligned = NERDataset.tokenize_and_align_labels(sample['words'], sample['ner'], student_tokenizer,
                                                                     True)

        teacher_input_ids_batch.append(teacher_input_aligned['input_ids'])
        student_input_ids_batch.append(student_input_aligned['input_ids'])
        teacher_att_mask_batch.append(teacher_input_aligned['attention_mask'])
        student_att_mask_batch.append(student_input_aligned['attention_mask'])
        student_labels_batch.append(student_input_aligned['label'])

    teacher_input_ids_batch = create_batch(teacher_input_ids_batch)
    student_input_ids_batch = create_batch(student_input_ids_batch)
    teacher_att_mask_batch = create_batch(teacher_att_mask_batch)
    student_att_mask_batch = create_batch(student_att_mask_batch)
    student_labels_batch = create_batch(student_labels_batch)

    return {
        'teacher_ids': teacher_input_ids_batch,
        'teacher_mask': teacher_att_mask_batch,
        'student_ids': student_input_ids_batch,
        'student_mask': student_att_mask_batch,
        'student_label': student_labels_batch
    }


def train(distiller, learning_rate, epochs, batch_size, weight_decay):
    wandb.init(project='rembert', entity='aitakaitov', tags=[h], config={
        'lr': args.lr,
        'batch_size': args.batch_size
    })

    dataset = load_dataset('json', data_files='polyglot_processed.json')['train']
    split_dataset = dataset.train_test_split(test_size=args.test_split_size, shuffle=True, seed=DATASET_SEED)
    train_dataset, test_dataset = split_dataset['train'], split_dataset['test']

    train_loader = DataLoader(train_dataset,
                              collate_fn=lambda x: x, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, collate_fn=lambda x: x)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    distiller.student_model = distiller.student_model.to(device)
    distiller.teacher_model = distiller.teacher_model.to(device)

    optimizer = AdamW(distiller.student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(epochs):
        distiller.student_model.train()
        for batch in train_loader:
            _input = preprocess_samples(batch)
            optimizer.zero_grad()
            student_preds, loss = distiller(_input['teacher_ids'].to(device), _input['teacher_mask'].to(device),
                                            _input['student_ids'].to(device), _input['student_mask'].to(device),
                                            _input['student_label'].to(device))
            loss.backward()
            optimizer.step()
            wandb.log({'train_loss': loss.item()})

        distiller.student_model.eval()
        predictions = []
        labels = []
        for batch in test_loader:
            _input = preprocess_samples(batch)
            labels.append(_input['student_label'].to('cpu').detach().numpy())
            student_preds = distiller(_input['teacher_ids'].to(device), _input['teacher_mask'].to(device),
                                      _input['student_ids'].to(device), _input['student_mask'].to(device),
                                      _input['student_label'].to(device), True)
            predictions.append(torch.squeeze(student_preds).to('cpu').detach().numpy())

        compute_metrics((np.array(predictions), labels))

    torch.save(distiller.student_model, h)


metric = load_metric("seqeval")

student_model, student_tokenizer = load_model(args.student_model, len(label_list))
teacher_model, teacher_tokenizer = load_model(args.teacher_model, len(label_list))

STUDENT_MAX_LENGTH = student_model.config.max_position_embeddings
TEACHER_MAX_LENGTH = student_model.config.max_position_embeddings

distiller = Distiller(student_model, teacher_model, args.temperature, args.loss)

train(distiller, args.lr, args.epochs, args.batch_size, args.weight_decay)
