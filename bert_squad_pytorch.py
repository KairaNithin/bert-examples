import json
import os
import re
import sys

import requests
import string
import numpy as np
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

gpu = torch.device('cuda')
# ============================================= DOWNLOADING DATA =======================================================
max_seq_length = 384
batch_size = 16
epochs = 4
train_data = requests.get("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json")
if train_data.status_code in (200,):
    with open('train.json', 'wb') as train_file:
        train_file.write(train_data.content)
eval_data = requests.get("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json")
if eval_data.status_code in (200,):
    with open('eval.json', 'wb') as eval_file:
        eval_file.write(eval_data.content)
with open('train.json') as f:
    raw_train_data = json.load(f)
with open('eval.json') as f:
    raw_eval_data = json.load(f)
slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
if not os.path.exists("bert_base_uncased/"):
    os.makedirs("bert_base_uncased/")
slow_tokenizer.save_pretrained("bert_base_uncased/")
tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)


# ============================================= PREPARING DATASET ======================================================


class Sample:
    def __init__(self, question, context, start_char_idx=None, answer_text=None, all_answers=None):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False
        self.start_token_idx = -1
        self.end_token_idx = -1

    def preprocess(self):
        context = " ".join(str(self.context).split())
        question = " ".join(str(self.question).split())
        tokenized_context = tokenizer.encode(context)
        tokenized_question = tokenizer.encode(question)
        if self.answer_text is not None:
            answer = " ".join(str(self.answer_text).split())
            end_char_idx = self.start_char_idx + len(answer)
            if end_char_idx >= len(context):
                self.skip = True
                return
            is_char_in_ans = [0] * len(context)
            for idx in range(self.start_char_idx, end_char_idx):
                is_char_in_ans[idx] = 1
            ans_token_idx = []
            for idx, (start, end) in enumerate(tokenized_context.offsets):
                if sum(is_char_in_ans[start:end]) > 0:
                    ans_token_idx.append(idx)
            if len(ans_token_idx) == 0:
                self.skip = True
                return
            self.start_token_idx = ans_token_idx[0]
            self.end_token_idx = ans_token_idx[-1]
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(tokenized_question.ids[1:])
        attention_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:
            self.skip = True
            return
        self.input_word_ids = input_ids
        self.input_type_ids = token_type_ids
        self.input_mask = attention_mask
        self.context_token_to_char = tokenized_context.offsets


def create_squad_examples(raw_data):
    squad_examples = []
    for item in raw_data["data"]:
        for para in item["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question = qa["question"]
                if "answers" in qa:
                    answer_text = qa["answers"][0]["text"]
                    all_answers = [_["text"] for _ in qa["answers"]]
                    start_char_idx = qa["answers"][0]["answer_start"]
                    squad_eg = Sample(question, context, start_char_idx, answer_text, all_answers)
                else:
                    squad_eg = Sample(question, context)
                squad_eg.preprocess()
                squad_examples.append(squad_eg)
    return squad_examples


def create_inputs_targets(squad_examples):
    dataset_dict = {
        "input_word_ids": [],
        "input_type_ids": [],
        "input_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in squad_examples:
        if item.skip is False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])
    x = [dataset_dict["input_word_ids"], dataset_dict["input_mask"], dataset_dict["input_type_ids"]]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y


def normalize_text(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    text = re.sub(regex, " ", text)
    text = " ".join(text.split())
    return text


train_squad_examples = create_squad_examples(raw_train_data)
x_train, y_train = create_inputs_targets(train_squad_examples)
print(f"{len(train_squad_examples)} training points created.")
eval_squad_examples = create_squad_examples(raw_eval_data)
x_eval, y_eval = create_inputs_targets(eval_squad_examples)
print(f"{len(eval_squad_examples)} evaluation points created.")

train_data = TensorDataset(torch.tensor(x_train[0], dtype=torch.int64),
                           torch.tensor(x_train[1], dtype=torch.float),
                           torch.tensor(x_train[2], dtype=torch.int64),
                           torch.tensor(y_train[0], dtype=torch.int64),
                           torch.tensor(y_train[1], dtype=torch.int64))
train_sampler = RandomSampler(train_data)
train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

eval_data = TensorDataset(torch.tensor(x_eval[0], dtype=torch.int64),
                          torch.tensor(x_eval[1], dtype=torch.float),
                          torch.tensor(x_eval[2], dtype=torch.int64),
                          torch.tensor(y_eval[0], dtype=torch.int64),
                          torch.tensor(y_eval[1], dtype=torch.int64))
eval_sampler = SequentialSampler(eval_data)
validation_data_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
# ================================================ TRAINING MODEL ======================================================
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device=gpu)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = torch.optim.Adam(lr=1e-5, betas=(0.9, 0.98), eps=1e-9, params=optimizer_grouped_parameters)

# model.load_state_dict(torch.load("./weights_4.pth"))

for epoch in range(1, epochs + 1):
    # ============================================ TRAINING ============================================================
    print("Training epoch ", str(epoch))
    training_pbar = tqdm(total=len(train_squad_examples), position=0, leave=True, file=sys.stdout)
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    for step, batch in enumerate(train_data_loader):
        batch = tuple(t.to(gpu) for t in batch)
        input_word_ids, input_mask, input_type_ids, start_token_idx, end_token_idx = batch
        optimizer.zero_grad()
        loss, _, _ = model(input_ids=input_word_ids,
                           attention_mask=input_mask,
                           token_type_ids=input_type_ids,
                           start_positions=start_token_idx,
                           end_positions=end_token_idx)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        nb_tr_steps += 1
        training_pbar.update(input_word_ids.size(0))
    training_pbar.close()
    print(f"\nTraining loss={tr_loss / nb_tr_steps:.4f}")
    torch.save(model.state_dict(), "./weights_" + str(epoch) + ".pth")
    # ============================================ VALIDATION ==========================================================
    validation_pbar = tqdm(total=len(eval_squad_examples), position=0, leave=True, file=sys.stdout)
    model.eval()
    eval_examples_no_skip = [_ for _ in eval_squad_examples if _.skip is False]
    currentIdx = 0
    count = 0
    for batch in validation_data_loader:
        batch = tuple(t.to(gpu) for t in batch)
        input_word_ids, input_mask, input_type_ids, start_token_idx, end_token_idx = batch
        with torch.no_grad():
            start_logits, end_logits = model(input_ids=input_word_ids,
                                             attention_mask=input_mask,
                                             token_type_ids=input_type_ids)
            pred_start, pred_end = start_logits.detach().cpu().numpy(), end_logits.detach().cpu().numpy()

        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            squad_eg = eval_examples_no_skip[currentIdx]
            currentIdx += 1
            offsets = squad_eg.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            if start >= len(offsets):
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = squad_eg.context[pred_char_start:pred_char_end]
            else:
                pred_ans = squad_eg.context[pred_char_start:]
            normalized_pred_ans = normalize_text(pred_ans)
            normalized_true_ans = [normalize_text(_) for _ in squad_eg.all_answers]
            if normalized_pred_ans in normalized_true_ans:
                count += 1
        validation_pbar.update(input_word_ids.size(0))
    acc = count / len(y_eval[0])
    validation_pbar.close()
    print(f"\nEpoch={epoch}, exact match score={acc:.2f}")

# ============================================ TESTING =================================================================
data = {"data":
    [
        {"title": "Project Apollo",
         "paragraphs": [
             {
                 "context": "The Apollo program, also known as Project Apollo, was the third United States human "
                            "spaceflight program carried out by the National Aeronautics and Space Administration ("
                            "NASA), which accomplished landing the first humans on the Moon from 1969 to 1972. First "
                            "conceived during Dwight D. Eisenhower's administration as a three-man spacecraft to "
                            "follow the one-man Project Mercury which put the first Americans in space, Apollo was "
                            "later dedicated to President John F. Kennedy's national goal of landing a man on the "
                            "Moon and returning him safely to the Earth by the end of the 1960s, which he proposed in "
                            "a May 25, 1961, address to Congress. Project Mercury was followed by the two-man Project "
                            "Gemini. The first manned flight of Apollo was in 1968. Apollo ran from 1961 to 1972, "
                            "and was supported by the two man Gemini program which ran concurrently with it from 1962 "
                            "to 1966. Gemini missions developed some of the space travel techniques that were "
                            "necessary for the success of the Apollo missions. Apollo used Saturn family rockets as "
                            "launch vehicles. Apollo/Saturn vehicles were also used for an Apollo Applications "
                            "Program, which consisted of Skylab, a space station that supported three manned missions "
                            "in 1973-74, and the Apollo-Soyuz Test Project, a joint Earth orbit mission with the "
                            "Soviet Union in 1975.",
                 "qas": [
                     {"question": "What project put the first Americans into space?",
                      "id": "Q1"
                      },
                     {"question": "What program was created to carry out these projects and missions?",
                      "id": "Q2"
                      },
                     {"question": "What year did the first manned Apollo flight occur?",
                      "id": "Q3"
                      },
                     {"question": "What President is credited with the original notion of putting Americans in space?",
                      "id": "Q4"
                      },
                     {"question": "Who did the U.S. collaborate with on an Earth orbit mission in 1975?",
                      "id": "Q5"
                      },
                     {"question": "How long did Project Apollo run?",
                      "id": "Q6"
                      },
                     {"question": "What program helped develop space travel techniques that Project Apollo used?",
                      "id": "Q7"
                      },
                     {"question": "What space station supported three manned missions in 1973-1974?",
                      "id": "Q8"
                      }
                 ]}]}]}

model.eval()
test_samples = create_squad_examples(data)
x_test, _ = create_inputs_targets(test_samples)
pred_start, pred_end = model(torch.tensor(x_test[0], dtype=torch.int64, device=gpu),
                             torch.tensor(x_test[1], dtype=torch.float, device=gpu),
                             torch.tensor(x_test[2], dtype=torch.int64, device=gpu))
pred_start, pred_end = pred_start.detach().cpu().numpy(), pred_end.detach().cpu().numpy()
for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
    test_sample = test_samples[idx]
    offsets = test_sample.context_token_to_char
    start = np.argmax(start)
    end = np.argmax(end)
    pred_ans = None
    if start >= len(offsets):
        continue
    pred_char_start = offsets[start][0]
    if end < len(offsets):
        pred_ans = test_sample.context[pred_char_start:offsets[end][1]]
    else:
        pred_ans = test_sample.context[pred_char_start:]
    print("Q: " + test_sample.question)
    print("A: " + pred_ans)
