import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, logging
from base_summary import SummaryGenerator

logging.set_verbosity_error()


class T5Generator(SummaryGenerator):
    """T5 Generative Summary Generator"""

    def __init__(self, model_path, device=None):
        super().__init__(model_path, device)
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=True)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
            self.model.eval()  # Switch to evaluation mode

        except Exception as e:
            raise RuntimeError(f"The T5 model failed to load: {str(e)}")

    def generate_summary(self, text, max_length=150, min_length=50, **kwargs):
        """"Generative summarization (Reorganizing language to generate new sentences)"""
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return "The input text is empty or invalid"

        # Build T5 task instructions (summarize task prefixes)
        input_text = f"summarize: {processed_text}"

        # Encoded input text
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="longest"
        ).to(self.device)

        # Generate parameter configuration
        generate_params = {
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": 8,
            "repetition_penalty": 1.5,
            "no_repeat_ngram_size": 3, **kwargs
        }

        # status summarizer
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generate_params
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)