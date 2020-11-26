import os

import torch
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, BertForQuestionAnswering

gpu = torch.device('cuda')

slow_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
if not os.path.exists("bert_base_cased/"):
    os.makedirs("bert_base_cased/")
slow_tokenizer.save_pretrained("bert_base_cased/")
tokenizer = BertWordPieceTokenizer("bert_base_cased/vocab.txt", lowercase=False)
model = BertForQuestionAnswering.from_pretrained('bert-large-cased-whole-word-masking-finetuned-squad').to(device=gpu)

context = "The Apollo program, also known as Project Apollo, was the third United States human spaceflight " \
          "program carried out by the National Aeronautics and Space Administration (NASA), which accomplished " \
          "landing the first humans on the Moon from 1969 to 1972. First conceived during Dwight D. " \
          "Eisenhower's administration as a three-man spacecraft to follow the one-man Project Mercury which " \
          "put the first Americans in space, Apollo was later dedicated to President John F. Kennedy's " \
          "national goal of landing a man on the Moon and returning him safely to the Earth by the end of the " \
          "1960s, which he proposed in a May 25, 1961, address to Congress. Project Mercury was followed by " \
          "the two-man Project Gemini. The first manned flight of Apollo was in 1968. Apollo ran from 1961 to " \
          "1972, and was supported by the two-man Gemini program which ran concurrently with it from 1962 to " \
          "1966. Gemini missions developed some of the space travel techniques that were necessary for the " \
          "success of the Apollo missions. Apollo used Saturn family rockets as launch vehicles. Apollo/Saturn " \
          "vehicles were also used for an Apollo Applications Program, which consisted of Skylab, " \
          "a space station that supported three manned missions in 1973-74, and the Apollo-Soyuz Test Project, " \
          "a joint Earth orbit mission with the Soviet Union in 1975. "

Q1 = "What project put the first Americans into space?"
Q2 = "What program was created to carry out these projects and missions?"
Q3 = "What year did the first manned Apollo flight occur?"
Q4 = "What President is credited with the original notion of putting Americans in space?"
Q5 = "Who did the U.S. collaborate with on an Earth orbit mission in 1975?"
Q6 = "How long did Project Apollo run?"
Q7 = "What program helped develop space travel techniques that Project Apollo used?"
Q8 = "What space station supported three manned missions in 1973-1974?"

questions = [Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8]

input_ids_all = []
token_type_ids_all = []
attention_masks = []
input_offsets = []
for question in questions:
    tokenized_context = tokenizer.encode(context)
    tokenized_question = tokenizer.encode(question)
    input_ids = tokenized_context.ids + tokenized_question.ids[1:]
    token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(tokenized_question.ids[1:])
    attention_mask = [1] * len(input_ids)
    padding_length = 384 - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
    input_ids_all.append(input_ids)
    token_type_ids_all.append(token_type_ids)
    attention_masks.append(attention_mask)
    input_offsets.append(tokenized_context.offsets)

start_scores, end_scores = model(input_ids=torch.tensor(input_ids_all, device=gpu),
                                 attention_mask=torch.tensor(attention_masks, device=gpu),
                                 token_type_ids=torch.tensor(token_type_ids_all, device=gpu))

# Q: What project put the first Americans into space?
# A: Project Mercury
# Q: What program was created to carry out these projects and missions?
# A: The Apollo program, also known as Project Apollo
# Q: What year did the first manned Apollo flight occur?
# A: 1968
# Q: What President is credited with the original notion of putting Americans in space?
# A: John F. Kennedy
# Q: Who did the U.S. collaborate with on an Earth orbit mission in 1975?
# A: Soviet Union
# Q: How long did Project Apollo run?
# A: 1961 to 1972
# Q: What program helped develop space travel techniques that Project Apollo used?
# A: Gemini missions
# Q: What space station supported three manned missions in 1973-1974?
# A: Skylab

for idx, (start, end) in enumerate(zip(start_scores, end_scores)):
    offsets = input_offsets[idx]
    print("Q:", questions[idx])
    print("A:", context[offsets[torch.argmax(start)][0]:offsets[torch.argmax(end)][1]])
