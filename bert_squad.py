import json
import os
import re
import string

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# ============================================= PREPARING DATASET ======================================================
train_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
train_path = keras.utils.get_file("train.json", train_data_url)
eval_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
eval_path = keras.utils.get_file("eval.json", eval_data_url)
with open(train_path) as f: raw_train_data = json.load(f)
with open(eval_path) as f: raw_eval_data = json.load(f)
max_seq_length = 384


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
            start_token_idx = ans_token_idx[0]
            end_token_idx = ans_token_idx[-1]
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
        if self.answer_text is not None:
            self.start_token_idx = start_token_idx
            self.end_token_idx = end_token_idx
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
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])
    x = [dataset_dict["input_word_ids"],
         dataset_dict["input_mask"],
         dataset_dict["input_type_ids"]]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y


# =================================================== TRAINING =========================================================

input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, input_type_ids])
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy().decode("utf-8")
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertWordPieceTokenizer(vocab=vocab_file, lowercase=True)
train_squad_examples = create_squad_examples(raw_train_data)
x_train, y_train = create_inputs_targets(train_squad_examples)
print(f"{len(train_squad_examples)} training points created.")
eval_squad_examples = create_squad_examples(raw_eval_data)
x_eval, y_eval = create_inputs_targets(eval_squad_examples)
print(f"{len(eval_squad_examples)} evaluation points created.")
start_logits = layers.Dense(1, name="start_logit", use_bias=False)(sequence_output)
start_logits = layers.Flatten()(start_logits)
end_logits = layers.Dense(1, name="end_logit", use_bias=False)(sequence_output)
end_logits = layers.Flatten()(end_logits)
start_probs = layers.Activation(keras.activations.softmax)(start_logits)
end_probs = layers.Activation(keras.activations.softmax)(end_logits)
model = keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=[start_probs, end_probs])
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = keras.optimizers.Adam(lr=5e-5)
model.compile(optimizer=optimizer, loss=[loss, loss])


def normalize_text(text):
    text = text.lower()
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    text = re.sub(regex, " ", text)
    text = " ".join(text.split())
    return text


class ValidationCallback(keras.callbacks.Callback):

    def __init__(self, x_eval, y_eval):
        self.x_eval = x_eval
        self.y_eval = y_eval

    def on_epoch_end(self, epoch, logs=None):
        pred_start, pred_end = self.model.predict(self.x_eval)
        count = 0
        eval_examples_no_skip = [_ for _ in eval_squad_examples if _.skip == False]
        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            squad_eg = eval_examples_no_skip[idx]
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
        acc = count / len(self.y_eval[0])
        print(f"\nepoch={epoch + 1}, exact match score={acc:.2f}")


model.fit(x_train, y_train, epochs=3, batch_size=8, callbacks=[ValidationCallback(x_eval, y_eval)])
model.save_weights("./weights.h5")
#model.load_weights("./weights.h5")

# ==================================================== TESTING =========================================================
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

test_context = create_squad_examples(data)
squad_test = test_context[0]
x_test, _ = create_inputs_targets(test_context)
pred_start, pred_end = model.predict(x_test)
for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
    offsets = squad_test.context_token_to_char
    start = np.argmax(start)
    end = np.argmax(end)
    pred_ans = None
    if start >= len(offsets):
        continue
    pred_char_start = offsets[start][0]
    if end < len(offsets):
        pred_ans = squad_test.context[pred_char_start:offsets[end][1]]
    else:
        pred_ans = squad_test.context[pred_char_start:]
    print("Q: " + test_context[idx].question)
    print("A: " + pred_ans)
