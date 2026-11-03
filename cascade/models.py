"""Small clients for the three model tasks; model services are started explicitly."""

import os

import numpy as np


class Embedder:
    def __init__(self, settings):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(settings["model"], device=settings["device"])
        self.model.max_seq_length = settings["max_length"]
        self.batch_size = settings["batch_size"]

    def encode(self, texts, *, query=False):
        prompt = "Instruct: Given a predicate, retrieve relevant evidence from the document\nQuery: " if query else ""
        return np.asarray(self.model.encode(
            texts, prompt=prompt, batch_size=self.batch_size,
            normalize_embeddings=True, show_progress_bar=False,
        ), dtype=np.float32)


class Reranker:
    def __init__(self, settings):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            settings["model"], device=settings["device"], max_length=settings["max_length"],
        )
        self.batch_size = settings["batch_size"]

    def score(self, query, chunks):
        import torch

        return self.model.predict(
            [(query, chunk) for chunk in chunks], batch_size=self.batch_size,
            activation_fn=torch.nn.Sigmoid(), show_progress_bar=False,
        ).tolist()


class ChatModel:
    def __init__(self, settings):
        from openai import OpenAI

        self.settings = settings
        key = os.environ.get(settings["api_key_env"], "")
        if settings["api_key_env"] != "SLM_API_KEY" and not key:
            raise ValueError(f"Set {settings['api_key_env']} before calling this model")
        self.client = OpenAI(base_url=settings["base_url"], api_key=key or "EMPTY", max_retries=0)

    def complete(self, messages, *, max_tokens, label_token_ids=None):
        options = dict(self.settings.get("request", {}))
        extra = dict(options.pop("extra_body", {}))
        if label_token_ids is not None:
            options.update(logprobs=True, top_logprobs=0)
            extra["logprob_token_ids"] = label_token_ids
        return self.client.chat.completions.create(
            model=self.settings["model"], messages=messages, max_completion_tokens=max_tokens,
            extra_body=extra or None, **options,
        )
