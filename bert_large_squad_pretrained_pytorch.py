import torch
from transformers import BertTokenizer, BertForQuestionAnswering

gpu = torch.device('cuda')
# Device:  cuda
print("Device:", gpu, "name:", torch.cuda.get_device_name(0))
# 1
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad').to(device=gpu)
# 2
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
# 3
input_ids = tokenizer.encode(input_text)
token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
# 4
start_scores, end_scores = model(torch.tensor([input_ids], device=gpu),
                                 token_type_ids=torch.tensor([token_type_ids], device=gpu))
# 5
all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
# a nice puppet
print(' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1]))
