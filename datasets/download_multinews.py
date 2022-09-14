"""
Multi-News: A Large-Scale Multi-Document Summarization Dataset and Abstractive Hierarchical Model, ACL 2019.

First download data from Huggingface and then use this script to prepare it in the required format.
"""

import jsonlines
from datasets import load_dataset


def loading_original_multinews():
    samples = []
    multinews_train = load_dataset("multi_news", split="train", ignore_verifications=True, cache_dir='~/mn')
    print("train", len(multinews_train))
    for s in multinews_train:
        samples.append(
            {"label": "train", "summary": s["summary"], "source_documents": s["document"].split('|||||')})

    multinews_val = load_dataset("multi_news", split="validation", ignore_verifications=True, cache_dir='~/mn')
    print("val", len(multinews_val))
    for s in multinews_val:
        samples.append(
            {"label": "val", "summary": s["summary"], "source_documents": s["document"].split('|||||')})

    multinews_test = load_dataset("multi_news", split="test", ignore_verifications=True)
    print("test", len(multinews_test))
    for s in multinews_test:
        samples.append(
            {"label": "test", "summary": s["summary"], "source_documents": s["document"].split('|||||')})

    return samples


if __name__ == "__main__":
    samples = loading_original_multinews()
    with jsonlines.open("multinews.jsonl", "w") as writer:
        writer.write_all(samples)
