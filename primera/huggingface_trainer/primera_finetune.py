#!/usr/bin/env python3
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, LEDTokenizer, LEDConfig, LEDForConditionalGeneration
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import wandb
wandb.login()


import sys
sys.path.append("../../")
from utils.metrics import rouge_corpus

data_path = "../../datasets/"

dataset_name = "multinews"
model_name = "allenai/PRIMERA"
wandb.init(project="PRIMERA_HT_%s_4096_1024"%dataset_name)

# load dataset
dataset_all = load_dataset('json', data_files=data_path + '%s.jsonl' % dataset_name, split='all')
print("dataset all", len(dataset_all))

dataset_train = dataset_all.filter(lambda s: s['label'] == 'train')
# dataset_train = dataset_train.shuffle(seed=42).select(range(128))
print("dataset train", len(dataset_train))

dataset_val = dataset_all.filter(lambda s: s['label'] == 'val')
# dataset_val = dataset_val.shuffle(seed=42).select(range(128))
print("dataset validation", len(dataset_val))

dataset_test = dataset_all.filter(lambda s: s['label'] == 'test')
# dataset_test = dataset_test.shuffle(seed=42).select(range(512))
print("dataset test selected", len(dataset_test))

# load tokenizer
tokenizer = LEDTokenizer.from_pretrained(model_name)
config = LEDConfig.from_pretrained(model_name)
# print(config)
config.gradient_checkpointing = False
config.num_beams = 5
config.max_length = 1024
config.min_length = 0
config.length_penalty = 2.0
config.early_stopping = True
config.no_repeat_ngram_size = 3

primera = LEDForConditionalGeneration.from_pretrained(model_name, config=config)

# training parameters
max_input_length = 4096
max_output_length = 1024
batch_size = 1


def process_data_to_model_inputs(batch):
    document_clusters = []
    for source_documents in batch["source_documents"]:
        max_length_doc = max_input_length // len(source_documents)
        input_text = []
        for source_document in source_documents:
            # preprocessing
            source_document = source_document.replace("\n", " ")
            source_document = " ".join(source_document.split())

            length = 0
            all_sents = sent_tokenize(source_document)
            for s in all_sents:
                input_text.append(s)
                length += len(s.split())
                if length >= max_length_doc:
                    break
            input_text.append("<doc-sep>")
        document_clusters.append(" ".join(input_text))
    # tokenize the inputs and labels
    summaries = batch["summary"]

    input_dict = tokenizer(document_clusters, padding='max_length', max_length=max_input_length,
                       truncation=True)
    outputs = tokenizer(
        summaries,
        padding="max_length",
        truncation=True,
        max_length=max_output_length
    )

    results = {}
    results["input_ids"] = input_dict.input_ids
    results["attention_mask"] = input_dict.attention_mask
    global_attention_tokens = [tokenizer.bos_token, tokenizer.additional_special_tokens[0]]
    batch["global_attention_mask"] = [[1 if token_id in tokenizer.convert_tokens_to_ids(global_attention_tokens) else 0 for token_id in input_ids_cluster] for input_ids_cluster in input_dict.input_ids]


    labels = outputs.input_ids
    # We have to make sure that the PAD token is ignored
    results["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in ls]
        for ls in labels
    ]

    return results


print("Preprocessing dataset train")
dataset_train = dataset_train.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size
)

print("Preprocessing dataset validation")
dataset_val = dataset_val.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size
)

print("Preprocessing dataset test")
dataset_test = dataset_test.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size
)

# set Python list to PyTorch tensor
dataset_train.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

# set Python list to PyTorch tensor
dataset_val.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

# set Python list to PyTorch tensor
dataset_test.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

training_args = Seq2SeqTrainingArguments(
    do_train=True,
    do_eval=True,
    do_predict=True,
    predict_with_generate=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    output_dir="result",
    logging_strategy="steps",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    warmup_steps=1000,
    save_total_limit=3,
    gradient_accumulation_steps=8,
    num_train_epochs=50,
    report_to='wandb'
)


# compute Rouge score during validation
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge_corpus(references=label_str, candidates=pred_str,
                                types=["rouge1", "rouge2", "rougeL", "rougeLsum"])

    result_dict = {
        "rouge1_fmeasure": round(rouge_output["rouge1"]["fmeasure"], 4),
        "rouge2_fmeasure": round(rouge_output["rouge2"]["fmeasure"], 4),
        "rougeL_fmeasure": round(rouge_output["rougeL"]["fmeasure"], 4),
        "rougeLsum_fmeasure": round(rouge_output["rougeLsum"]["fmeasure"], 4),
    }
    print(result_dict)
    return result_dict


# instantiate trainer
trainer = Seq2SeqTrainer(
    model=primera,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset_train,
    eval_dataset=dataset_val
)

# testing
trainer.predict(test_dataset=dataset_test)
# training
trainer.train()
