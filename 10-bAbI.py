import os


def parse_babi_file(file_path):
    """
    Split bAbI txt and return list of episodes:
    [{'story': ..., 'question': ..., 'answer': ...}, ...]
    """
    examples = []
    story_lines = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Remove sentence number
            idx, text = line.split(' ', 1)
            idx = int(idx)

            if '\t' in text:  # check marker of question
                question, answer, _ = text.split('\t')
                # Construct prompt: whole history before question
                story = ' '.join(story_lines)
                examples.append({
                    'story': story,
                    'question': question,
                    'answer': answer
                })
            else:
                story_lines.append(text)

            # Reset history by new episode (new marker == 1)
            if idx == 1:
                story_lines = [text]
    return examples


if __name__ == "__main__":

    file_path = "datasets/bAbI/en-10k/qa1_single-supporting-fact_train.txt"
    items = parse_babi_file(file_path)

    prompts_targets = []
    for i, item in enumerate(items):
        if i >= 5: break
        print(f"### Example {i + 1}:")
        print("# Story:", item["story"])
        print("# Question:", item["question"])
        print("# Answer:", item["answer"])
        print()
