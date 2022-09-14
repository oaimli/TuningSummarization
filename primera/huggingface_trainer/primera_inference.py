import numpy as np
import random
import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration
from datasets import load_dataset
from nltk.tokenize import sent_tokenize

import sys
sys.path.append('../../')
from utils.metrics import rouge_corpus


device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path = "../../datasets/"
dataset_name = "multinews"
model_name = "allenai/PRIMERA-multinews"
max_input_length = 4096
max_output_length = 1024
beam_size = 5

# load dataset
dataset_all = load_dataset('json', data_files=data_path + '%s.jsonl' % dataset_name, split='all')
print("dataset all", len(dataset_all))
dataset_test = dataset_all.filter(lambda s: s['label'] == 'test')
dataset_test = dataset_test.shuffle(seed=42).select(range(36))
print("dataset test selected", len(dataset_test))

tokenizer = LEDTokenizer.from_pretrained(model_name)
model = LEDForConditionalGeneration.from_pretrained(model_name).to(device)
# model.gradient_checkpointing_enable()
docsep_token_id = tokenizer.convert_tokens_to_ids("<doc-sep>")

def generate_answer(batch):
    source_clusters = []
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
        source_clusters.append(" ".join(input_text))
    inputs_dict = tokenizer(source_clusters, padding="max_length", max_length=max_input_length, return_tensors="pt", truncation=True)
    input_ids = inputs_dict.input_ids.to("cuda")
    attention_mask = inputs_dict.attention_mask.to("cuda")
    global_attention_mask = torch.zeros_like(attention_mask)
    # put global attention on <s> and <doc-sep> token
    global_attention_mask[:, 0] = 1
    global_attention_mask[input_ids == docsep_token_id] = 1

    predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask,  max_length=max_output_length, num_beams=beam_size)
    batch["predicted_summary"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
    return batch


results = dataset_test.map(generate_answer, batched=True, batch_size=4)


print("model %s"%model_name)
scores = rouge_corpus(references=results["summary"], candidates=results["predicted_summary"], types=["rouge1", "rouge2", "rougeL", "rougeLsum"])
print("Inference result on Multi-news:")
print("rouge-1", scores["rouge1"]["fmeasure"])
print("rouge-2", scores["rouge2"]["fmeasure"])
print("rouge-L", scores["rougeL"]["fmeasure"])
print("rouge-Lsum", scores["rougeLsum"]["fmeasure"])

data_idx = random.choices(range(len(results)), k=10)
for item in results.select(data_idx):
    print("####summary: ", item['summary'])
    print("####predicted_summary: ", item['predicted_summary'])