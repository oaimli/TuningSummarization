from torch.utils.data import DataLoader, Dataset
import torch
import random
from nltk.tokenize import sent_tokenize
from datasets import load_dataset, concatenate_datasets


class SummarizationDataset(Dataset):
    def __init__(
        self,
        dataset,
        dataset_name,
        documents_concatenation,
        tokenizer,
        max_input_len,
        max_output_len,
        mask_num=5,
        dataset_type="train",
    ):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.documents_concatenation = documents_concatenation
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        if documents_concatenation == "concat_start_wdoc_global":
            self.docsep_token_id = self.tokenizer.additional_special_tokens_ids[0]
        self.mask_id = self.tokenizer.mask_token_id
        self.mask_num = mask_num
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        # single doc setting
        all_docs = entry["source_documents"]
        tgt = entry["summary"]

        # pre-processing
        if self.dataset_name == "multinews":
            for i, doc in enumerate(all_docs):
                doc = doc.replace("\n", " ")
                doc = " ".join(doc.split())
                all_docs[i] = doc

        # combination of source documents
        if self.documents_concatenation == "plain_concat":
            src = "\n".join(all_docs)
            input_ids = self.tokenizer.encode(
                src, truncation=True, max_length=self.max_input_len
            )
        elif self.documents_concatenation == "concat_start_eachdoc":
            input_text = []
            for doc in all_docs:
                length = 0
                all_sents = sent_tokenize(doc)
                for s in all_sents:
                    input_text.append(s)
                    length += len(s.split())
                    if length >= self.max_input_len // len(all_docs):
                        break
            input_ids = self.tokenizer.encode(
                " ".join(input_text),
                truncation=True,
                max_length=self.max_input_len,
            )
        elif self.documents_concatenation == "concat_start_eachdoc_wsent_global":
            input_ids = []
            for doc in all_docs:
                sents = [
                    " [sent] ".join(sent_tokenize(p)) + " [sent]"
                    for p in doc.split("\n")
                    if p != ""
                ]
                doc = "\n".join(sents)
                input_ids.extend(
                    self.tokenizer.encode(
                        doc,
                        truncation=True,
                        max_length=self.max_input_len // len(all_docs),
                    )[1:-1]
                )
            input_ids = (
                [self.tokenizer.bos_token_id]
                + input_ids
                + [self.tokenizer.eos_token_id]
            )
        elif self.documents_concatenation == "concat_start_wdoc_global":
            mask_num = self.mask_num

            input_ids = [self.mask_id] * mask_num if mask_num > 0 else []
            for doc in all_docs:
                input_ids.extend(
                    self.tokenizer.encode(
                        doc,
                        truncation=True,
                        max_length=(self.max_input_len - mask_num) // len(all_docs),
                    )[1:-1]
                )
                input_ids.append(self.docsep_token_id)
            input_ids = (
                [self.tokenizer.bos_token_id]
                + input_ids
                + [self.tokenizer.eos_token_id]
            )

        output_ids = self.tokenizer.encode(
            tgt, truncation=True, max_length=self.max_output_len
        )

        if self.tokenizer.bos_token_id is None:  # pegasus
            output_ids = [self.tokenizer.pad_token_id] + output_ids
        if self.dataset_type == "train":
            return torch.tensor(input_ids), torch.tensor(output_ids)
        else:
            return torch.tensor(input_ids), torch.tensor(output_ids), tgt


def collate_fn(batch):
    # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level member variables
    if batch[0][0][-1].item() == 2:
        pad_token_id = (
            1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        )
    elif batch[0][0][-1].item() == 1:
        pad_token_id = (
            0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        )
    else:
        assert False
    train = True
    if len(batch[0]) == 3:
        train = False
        tgt = [item[2] for item in batch]
        batch = [item[:2] for item in batch]
    input_ids, output_ids = list(zip(*batch))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    output_ids = torch.nn.utils.rnn.pad_sequence(
        output_ids, batch_first=True, padding_value=pad_token_id
    )
    if train:
        return input_ids, output_ids
    else:
        return input_ids, output_ids, tgt


def get_dataloader_summ(
    args, tokenizer, split_name, num_workers, is_shuffle
):
    dataset_all = load_dataset('json', data_files=args.data_path + '%s.json' % args.dataset_name, split='all')
    print("%s all"%args.dataset_name, len(dataset_all))

    random.seed(args.rand_seed)# This is to control random selection of training and testing samples
    dataset = []
    if split_name == "train":
        dataset = dataset_all.filter(lambda s: s['label'] == 'train')
        print("dataset train all", len(dataset))
        if args.num_train_data != -1 and 0 < args.num_train_data < len(list(dataset)):
            dataset = dataset.select(random.choices(range(len(dataset)), k=args.num_train_data))
            print("dataset train selected", len(dataset))
    if split_name == "validation":
        dataset = dataset_all.filter(lambda s: s['label'] == 'val')
        if len(dataset)> args.num_val_data > 0:
            dataset = dataset.select(random.choices(range(len(dataset)), k=args.num_val_data))
        print("dataset validation", len(dataset))
    if split_name == "test":
        dataset = dataset_all.filter(lambda s: s['label'] == 'test')
        if len(dataset)> args.num_test_data > 0:
            dataset = dataset.select(random.choices(range(len(dataset)), k=args.num_test_data))
        print("dataset test selected", len(dataset))

    summarization_dataset = SummarizationDataset(
        dataset=dataset,
        dataset_name=args.dataset_name,
        documents_concatenation=args.documents_concatenation,
        tokenizer=tokenizer,
        max_input_len=args.max_length_input,
        max_output_len=args.max_length_tgt,
        mask_num=args.mask_num,
        dataset_type=split_name,
    )

    return DataLoader(
        summarization_dataset,
        batch_size=args.batch_size,
        shuffle=is_shuffle,
        num_workers=num_workers,
        # sampler=sampler,
        collate_fn=collate_fn,
    )