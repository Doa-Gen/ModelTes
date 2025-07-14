import numpy as np
import torch
from transformers import BertTokenizer, BertModel, logging
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize.punkt import PunktSentenceTokenizer
from base_summary import SummaryGenerator

logging.set_verbosity_error()


class BERTGenerator(SummaryGenerator):
    """BERT Extractive Summary Generator (Specify Local Segmental path)"""

    def __init__(self, model_path, punkt_tokenizer_path, device=None):

        super().__init__(model_path, device)

        # Load the local tokenizer
        try:
            self.sent_tokenizer = PunktSentenceTokenizer(punkt_tokenizer_path)
            print(f"The local tokenizer was successfully loaded: {punkt_tokenizer_path}")
        except Exception as e:
            raise RuntimeError(f"The tokenizer failed to load {str(e)}")

        # BERT
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertModel.from_pretrained(model_path).to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"The BERT model failed to load: {str(e)}")

    def generate_summary(self, text, num_sentences=3):
        """Generate extractive summaries"""
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return "The input text is empty or invalid"

        sentences = self.sent_tokenizer.tokenize(processed_text)

        summary_size = min(num_sentences, len(sentences))

        sentence_embeddings = self._get_sentence_embeddings(sentences)

        doc_vector = np.mean(sentence_embeddings, axis=0)

        similarities = cosine_similarity(
            sentence_embeddings,
            doc_vector.reshape(1, -1)
        ).flatten()

        top_indices = np.argsort(-similarities)[:summary_size]
        top_indices.sort()

        return " ".join([sentences[i] for i in top_indices])

    def _get_sentence_embeddings(self, sentences):
        """Obtain the BERT embedding vector of the sentence"""
        embeddings = []
        for sentence in sentences:
            inputs = self.tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings.append(cls_embedding)

        return np.array(embeddings)