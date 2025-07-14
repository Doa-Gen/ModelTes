import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from bert_summary import BERTGenerator
from t5_summary import T5Generator


def read_text_file(file_path):
    """Read the content of the text file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"File reading failed: {str(e)}")
        return None


def save_to_txt(content, filename):
    """Save the content to a text file"""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return file_path


def main():
    bert_model_path = r"D:\modelLib\bert-small"
    t5_model_path = r"D:\modelLib\T5-small"

    punkt_path = r"D:\Anaconda\envs\deepseek-r1\nltk_data\tokenizers\punkt\english.pickle"

    input_file = "read.txt"
    print(f"Read the input text: {input_file}")
    test_text = read_text_file(input_file)

    if not test_text:
        print("The input text is empty or the reading fails, and the program exits!")
        return

    print("\nInitialize the model:")
    try:
        bert_generator = BERTGenerator(bert_model_path, punkt_path)
        t5_generator = T5Generator(t5_model_path)
        print("The model initialization was successful!")
    except Exception as e:
        print(f"Model initialization failed!: {str(e)}")
        return

    print("\nStart generating the summary")

    try:
        # BERT
        print("Generate BERT summary...")
        bert_summary = bert_generator.generate_summary(test_text, num_sentences=3)

        # T5
        print("Generate BERT summary...")
        t5_summary = t5_generator.generate_summary(test_text, max_length=120, min_length=60)

        print("Generate BERT summary...")
    except Exception as e:
        print(f"Summary generation failed: {str(e)}")
        return

    output_content = f"""
Abstract generation result

original text:
{test_text}

BERT Extractive summary:
{bert_summary}

T5 Generative summary:
{t5_summary}

"""
    filename = "summary_result.txt"
    file_path = save_to_txt(output_content, filename)

    print(f"The summary has been saved to: {file_path}")

if __name__ == "__main__":
    main()