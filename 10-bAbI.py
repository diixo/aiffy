import os

def parse_babi_file(file_path):
    """
    Разбирает bAbI txt файл и возвращает список примеров:
    [{'story': ..., 'question': ..., 'answer': ...}, ...]
    """
    examples = []
    story_lines = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Убираем номер предложения
            idx, text = line.split(' ', 1)
            idx = int(idx)

            if '\t' in text:  # это вопрос
                question, answer, _ = text.split('\t')
                # Формируем prompt: вся история до вопроса
                story = ' '.join(story_lines)
                examples.append({
                    'story': story,
                    'question': question,
                    'answer': answer
                })
                #story_lines.append(question)  # можно добавлять вопрос в историю, если нужна память
            else:  # это факт истории
                story_lines.append(text)

            # Сброс истории при новом эпизоде (номер предложения == 1)
            if idx == 1:
                story_lines = [text]

    return examples


if __name__ == "__main__":

    file_path = "datasets/bAbI/en-10k/qa1_single-supporting-fact_train.txt"
    items = parse_babi_file(file_path)

    # Генерация prompts/targets для GPT-2
    prompts_targets = []
    for i, item in enumerate(items):
        if i >= 5: break
        print(f"### Example {i + 1}:")
        print("# Story:", item["story"])
        print("# Question:", item["question"])
        print("# Answer:", item["answer"])
        print()


