import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def load_babi_txt(file_path: str):
    """
    Split bAbI txt specified file and return list of episodes:
    [{'story': ..., 'question': ..., 'answer': ...}, ...]
    """
    examples = []
    story_lines = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # remove sentence number
            idx, text = line.split(' ', 1)
            idx = int(idx)

            if '\t' in text:  # check marker of question
                question, answer, _ = text.split('\t')
                # construct prompt: whole history before question
                story = ' '.join(story_lines)
                examples.append({
                    'story': story,
                    'question': question,
                    'answer': answer
                })
            else:
                story_lines.append(text)

            # reset history by new episode (new marker == 1)
            if idx == 1:
                story_lines = [text]
    return examples


def evaluate_gpt2_on_babi(file_path: str, max_new_tokens: int = 30):

    data = load_babi_txt(file_path)

    print("::evaluate_gpt2 " + 54 * "*")
    results = []

    for i, sample in enumerate(data):
        if (i % 100 == 0) and (i > 0):
            print("...item:", i)

        story = sample["story"]
        question = sample["question"]
        answer = sample["answer"]

        prompt = f"{story} {question}"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,  # жёсткая генерация (greedy)
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Answer:" in generated_text:
            pred = generated_text.split("Answer:")[-1].strip()
        else:
            pred = generated_text.strip()

        results.append({
            "story": story,
            "question": question,
            "true_answer": answer,
            "predicted_answer": pred
        })
    return results


if __name__ == "__main__":

    file_path = "datasets/bAbI/en-10k/qa1_single-supporting-fact_train.txt"
    items = load_babi_txt(file_path)

    prompts_targets = []
    for i, item in enumerate(items):
        if i >= 5: break
        print(f"### Example {i + 1}:")
        print(f"# Story: {item['story']}")
        print(f"# Question: {item['question']}")
        print(f"# Answer: {item['answer']}\n")

    ######################################################################

    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = evaluate_gpt2_on_babi(file_path)

    # Можно прикинуть точность (экзакт матч)
    correct = sum(r["true_answer"].lower() == r["predicted_answer"].lower()
                for r in results)
    print(f"Accuracy: {correct}/{len(results)} = {correct/len(results):.2%}")
