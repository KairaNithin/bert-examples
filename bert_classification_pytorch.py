import torch
from datasets import load_dataset
import numpy as np
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

epochs = 4
batch_size = 16
max_seq_length = 128
gpu = torch.device('cuda')
train_data = load_dataset('glue', 'mrpc', split='train')
eval_data = load_dataset('glue', 'mrpc', split='validation')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples):
    labels = []
    input_word_ids = []
    input_type_ids = []
    input_masks = []
    for (index, example) in enumerate(examples):
        if "label" in example:
            labels.append(example["label"])
        tokens_a = tokenizer.tokenize(example["sentence1"])
        tokens_b = tokenizer.tokenize(example["sentence2"])
        truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        input_word_ids.append(input_ids)
        input_type_ids.append(segment_ids)
        input_masks.append(input_mask)
    return [torch.tensor(input_word_ids, dtype=torch.int64),
            torch.tensor(input_masks, dtype=torch.float),
            torch.tensor(input_type_ids, dtype=torch.int64),
            torch.tensor(labels, dtype=torch.int64)]


train_data = TensorDataset(*convert_examples_to_features(train_data))
train_sampler = RandomSampler(train_data)
train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

eval_data = TensorDataset(*convert_examples_to_features(eval_data))
eval_sampler = SequentialSampler(eval_data)
validation_data_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

# ================================================ TRAINING MODEL ======================================================
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device=gpu)
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
    training_pbar = tqdm(total=len(train_data), position=0, leave=True)
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    for step, batch in enumerate(train_data_loader):
        batch = tuple(t.to(gpu) for t in batch)
        input_word_ids, input_mask, input_type_ids, labels = batch
        optimizer.zero_grad()
        loss, _ = model(input_ids=input_word_ids,
                        attention_mask=input_mask,
                        token_type_ids=input_type_ids,
                        labels=labels)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        nb_tr_steps += 1
        training_pbar.update(input_word_ids.size(0))
    training_pbar.close()
    print(f"\nTraining loss={tr_loss / nb_tr_steps:.4f}")
    torch.save(model.state_dict(), "./weights_" + str(epoch) + ".pth")
    # ============================================ VALIDATION ==========================================================
    validation_pbar = tqdm(total=len(eval_data), position=0, leave=True)
    model.eval()
    eval_accuracy = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in validation_data_loader:
        batch = tuple(t.to(gpu) for t in batch)
        input_word_ids, input_mask, input_type_ids, labels = batch
        with torch.no_grad():
            logits = model(input_ids=input_word_ids,
                           attention_mask=input_mask,
                           token_type_ids=input_type_ids)

        logits = logits[0].detach().cpu().numpy()
        label_ids = labels.cpu().numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        eval_accuracy += np.sum(pred_flat == labels_flat) / len(labels_flat)
        nb_eval_steps += 1
        validation_pbar.update(input_word_ids.size(0))
    validation_pbar.close()
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

# ============================================ TESTING =================================================================
model.eval()
input_word_ids_test, input_masks_test, input_type_ids_test, _ = convert_examples_to_features(
    [{
        'sentence1': 'The rain in Spain falls mainly on the plain.',
        'sentence2': 'It mostly rains on the flat lands of Spain.'
    }, {
        'sentence1': 'Look I fine tuned BERT.',
        'sentence2': 'Is it working? This does not match.'
    }])
result = model(input_ids=input_word_ids_test.to(gpu),
               attention_mask=input_masks_test.to(gpu),
               token_type_ids=input_type_ids_test.to(gpu))
result = result[0].detach().cpu()
print(result.numpy())
result = torch.argmax(result, dim=1).numpy()
print(result)
