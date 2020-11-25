import os

import torch
from datasets import load_dataset
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from transformers import BertTokenizer
max_seq_length = 128
slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
if not os.path.exists("bert_base_uncased/"):
    os.makedirs("bert_base_uncased/")
slow_tokenizer.save_pretrained("bert_base_uncased/")
tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)

batch_size = 16
epochs = 10
gpu = torch.device('cuda')
train_data = load_dataset('glue', 'mrpc', split='train')
eval_data = load_dataset('glue', 'mrpc', split='validation')

train_sampler = RandomSampler(train_data)
train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, tokenizer, max_seq_length):

    # label_map = {}
    # for (i, label) in enumerate(label_list):
    #     label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.encode(example.text_a)
        tokens_b = tokenizer.encode(example.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
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
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # label_id = label_map[example.label]
        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))
        #
        # features.append(
        #     InputFeatures(input_ids=input_ids,
        #                   input_mask=input_mask,
        #                   segment_ids=segment_ids,
        #                   label_id=label_id))
    return features


convert_examples_to_features(train_data, tokenizer, max_seq_length)
eval_sampler = SequentialSampler(eval_data)
validation_data_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

# # ================================================ TRAINING MODEL ======================================================
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device=gpu)
# param_optimizer = list(model.named_parameters())
# no_decay = ['bias', 'gamma', 'beta']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
#      'weight_decay_rate': 0.01},
#     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
#      'weight_decay_rate': 0.0}
# ]
#
# optimizer = torch.optim.Adam(lr=1e-5, betas=(0.9, 0.98), eps=1e-9, params=optimizer_grouped_parameters)
#
# # model.load_state_dict(torch.load("./weights_4.pth"))
#
# for epoch in range(1, epochs + 1):
#     # ============================================ TRAINING ============================================================
#     print("Training epoch ", str(epoch))
#     training_pbar = tqdm(total=len(train_squad_examples), position=0, leave=True)
#     model.train()
#     tr_loss = 0
#     nb_tr_examples, nb_tr_steps = 0, 0
#     for step, batch in enumerate(train_data_loader):
#         batch = tuple(t.to(gpu) for t in batch)
#         input_word_ids, input_mask, input_type_ids, start_token_idx, end_token_idx = batch
#         optimizer.zero_grad()
#         loss, _, _ = model(input_ids=input_word_ids,
#                            attention_mask=input_mask,
#                            token_type_ids=input_type_ids,
#                            start_positions=start_token_idx,
#                            end_positions=end_token_idx)
#         loss.backward()
#         optimizer.step()
#         tr_loss += loss.item()
#         nb_tr_examples += input_word_ids.size(0)
#         nb_tr_steps += 1
#         training_pbar.update(input_word_ids.size(0))
#     print(f"\nTraining loss={tr_loss / nb_tr_steps:.4f}")
#     training_pbar.close()
#     torch.save(model.state_dict(), "./weights_" + str(epoch) + ".pth")
#     # ============================================ VALIDATION ==========================================================
#     validation_pbar = tqdm(total=len(eval_squad_examples), position=0, leave=True)
#     model.eval()
#     eval_loss, eval_accuracy = 0, 0
#     nb_eval_steps, nb_eval_examples = 0, 0
#     eval_examples_no_skip = [_ for _ in eval_squad_examples if _.skip is False]
#     currentIdx = 0
#     count = 0
#     for batch in validation_data_loader:
#         batch = tuple(t.to(gpu) for t in batch)
#         input_word_ids, input_mask, input_type_ids, start_token_idx, end_token_idx = batch
#         with torch.no_grad():
#             start_logits, end_logits = model(input_ids=input_word_ids,
#                                              attention_mask=input_mask,
#                                              token_type_ids=input_type_ids)
#             pred_start, pred_end = start_logits.detach().cpu().numpy(), end_logits.detach().cpu().numpy()
#
#         for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
#             squad_eg = eval_examples_no_skip[currentIdx]
#             currentIdx += 1
#             offsets = squad_eg.context_token_to_char
#             start = np.argmax(start)
#             end = np.argmax(end)
#             if start >= len(offsets):
#                 continue
#             pred_char_start = offsets[start][0]
#             if end < len(offsets):
#                 pred_char_end = offsets[end][1]
#                 pred_ans = squad_eg.context[pred_char_start:pred_char_end]
#             else:
#                 pred_ans = squad_eg.context[pred_char_start:]
#             normalized_pred_ans = normalize_text(pred_ans)
#             normalized_true_ans = [normalize_text(_) for _ in squad_eg.all_answers]
#             if normalized_pred_ans in normalized_true_ans:
#                 count += 1
#         validation_pbar.update(input_word_ids.size(0))
#     acc = count / len(y_eval[0])
#     print(f"\nEpoch={epoch}, exact match score={acc:.2f}")
#     validation_pbar.close()
#
# # ============================================ TESTING =================================================================
