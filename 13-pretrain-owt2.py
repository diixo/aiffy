
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

    #ds = load_dataset("EleutherAI/pile", split="train") #= 825Gb

    #ds = load_dataset("EleutherAI/pile", name="openwebtext2", split="train", trust_remote_code=True)

    if True:

        ds = load_dataset("OpenAssistant/oasst1")
        print(">>> oasst1:", len(ds), 24 * "*")

        ds = load_dataset("blended_skill_talk")
        print(">>> blended_skill_talk:", len(ds), 24 * "*")

        # ds = load_dataset("topical_chat")

        # print(">>> topical_chat:", len(ds), 24 * "*")

        # looks like: https://huggingface.co/datasets/Skylion007/openwebtext
        ds = load_dataset("openwebtext", split="train", trust_remote_code=True)
        print(">>> openwebtext:", len(ds), 24 * "*")

        # ds.to_json("openwebtext2.jsonl", orient="records", lines=True)
