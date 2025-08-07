
"""
The Pile is a huge dataset (825 GB of text) created by EleutherAI as a corpus for pre-training large language models.
It consists of 15 different subsets, including:

📚 books3
🌐 openwebtext2
🧑‍🔬 pubmed
🧵 stackexchange
📄 arxiv
🧑‍🏫 wiki
📈 github
...
"""

from datasets import load_dataset


if __name__ == "__main__":

    # pile = load_dataset("EleutherAI/pile", split="train") = 825Gb

    ds = load_dataset("EleutherAI/pile", name="openwebtext2", split="train", trust_remote_code=True)

    print(len(ds))

    # ds.to_json("openwebtext2.jsonl", orient="records", lines=True)
