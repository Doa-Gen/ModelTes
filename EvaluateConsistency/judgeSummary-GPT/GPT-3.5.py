from openai import OpenAI
import re

client = OpenAI(
    api_key="",
    base_url=""
)

def read_summary_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_text = re.search(r'original text:(.*?)BERT Extractive summary:', content, re.DOTALL).group(1).strip()
    bert_summary = re.search(r'BERT Extractive summary:(.*?)T5 Generative summary:', content, re.DOTALL).group(
        1).strip()
    t5_summary = re.search(r'T5 Generative summary:(.*?)$', content, re.DOTALL).group(1).strip()

    original_text = re.sub(r'\s+', ' ', original_text).strip()
    bert_summary = re.sub(r'\s+', ' ', bert_summary).strip()
    t5_summary = re.sub(r'\s+', ' ', t5_summary).strip()

    return original_text, bert_summary, t5_summary


def gpt_evaluate(article, summary, prompt_type):
    if prompt_type == "zero-shot":
        prompt = f"""Please determine whether the provided summary is consistent with the corresponding article. Note that "consistency" refers to how much information included in the summary is present in the source article.
Article: {article}
Summary: {summary}
Answer: (yes or no)"""
    elif prompt_type == "chain-of-thoughts":
        prompt = f"""Please determine whether the provided summary is consistent with the corresponding article. Note that "consistency" refers to how much information included in the summary is present in the source article.
Article: {article}
Summary: {summary}b
Answer: Explain your reasoning step y step then answer the question (yes or no)"""
    elif prompt_type == "score":
        prompt = f"""Score the following summary given the corresponding article with respect to consistency from 0 to 1 where 1 means most consistent. Note that "consistency" refers to how much information included in the summary is present in the source article.
Article: {article}
Summary: {summary}
Score:"""
    else:
        raise ValueError("prompt_type must be 'zero-shot', 'chain-of-thoughts', or 'score'")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    file_path = "E:\python_works\BertSmall\summaryGroup\output\summary_result.txt"
    original_text, bert_summary, t5_summary = read_summary_file(file_path)

    print("Model(boltuix/bert-small) Evaluation:")
    print("1. Zero-shot Evaluation:")
    print(gpt_evaluate(original_text, bert_summary, "zero-shot"))

    print("\n2. Chain-of-thoughts Evaluation")
    print(gpt_evaluate(original_text, bert_summary, "chain-of-thoughts"))

    print("\n3. Score:")
    print(gpt_evaluate(original_text, bert_summary, "score"))

    print("\nModel(t5-small) Evaluation:")
    print("1. Zero-shot Evaluation:")
    print(gpt_evaluate(original_text, t5_summary, "zero-shot"))

    print("\n2. Chain-of-thoughts Evaluation:")
    print(gpt_evaluate(original_text, t5_summary, "chain-of-thoughts"))

    print("\n3. Score:")
    print(gpt_evaluate(original_text, t5_summary, "score"))