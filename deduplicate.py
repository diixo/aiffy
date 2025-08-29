
from pathlib import Path
from collections import Counter


def read_dictionary_set(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        word_set = set([line.strip() for line in f if line.strip()])
    print(f"[{filename}].sz={len(word_set)}")
    return word_set


def save_dictionary(word_set: set):

    word_list = sorted(word_set)

    path = Path("cached-tmp-diff.txt")
    with path.open("w", encoding="utf-8") as f:
        for word in word_list:
            f.write(word + "\n")

    print(f"Saved: file.sz={len(word_list)}")


def main():

    dictionary = read_dictionary_set("db-full.txt")

    diixonary = read_dictionary_set("db-full-current.txt")

    common = dictionary & diixonary

    #print(common)

    diff = dictionary ^ diixonary
    if len(diff) > 0:
        save_dictionary(diff)
        return

    # # summ = dictionary | diixonary

    db_full = read_dictionary_set("db-full.txt")

    print(len(db_full))


if __name__ == "__main__":
    main()
