"""Small clients for the three model tasks; model services are started explicitly."""

import os

import numpy as np


class Embedder:
    def __init__(self, model, *, device="cuda", batch_size=16, max_length=32768):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device=device)
        self.model.max_seq_length = max_length
        self.batch_size = batch_size

    def encode(self, texts, *, query=False):
        prompt = "Instruct: Given a predicate, retrieve relevant evidence from the document\nQuery: " if query else ""
        return np.asarray(self.model.encode(
            texts, prompt=prompt, batch_size=self.batch_size,
            normalize_embeddings=True, show_progress_bar=False,
        ), dtype=np.float32)


class Reranker:
    def __init__(self, model, *, device="cuda", batch_size=8, max_length=8192):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            model, device=device, max_length=max_length,
        )
        self.batch_size = batch_size

    def score(self, query, chunks):
        import torch

        return self.model.predict(
            [(query, chunk) for chunk in chunks], batch_size=self.batch_size,
            activation_fn=torch.nn.Sigmoid(), show_progress_bar=False,
        ).tolist()


class ChatModel:
    def __init__(self, model, base_url, *, api_key_env=None, temperature=0, seed=None,
                 reasoning_effort=None, disable_thinking=False):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.reasoning_effort = reasoning_effort
        self.disable_thinking = disable_thinking
        key = os.environ.get(api_key_env, "") if api_key_env else ""
        if api_key_env and not key:
            raise ValueError(f"Set the {api_key_env} environment variable before calling this model")
        self.client = OpenAI(base_url=base_url, api_key=key or "EMPTY", max_retries=0)

    def complete(self, messages, *, max_tokens, label_token_ids=None):
        options = {"temperature": self.temperature}
        extra = {}
        if self.seed is not None:
            options["seed"] = self.seed
        if self.reasoning_effort is not None:
            options["reasoning_effort"] = self.reasoning_effort
        if self.disable_thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        if label_token_ids is not None:
            options.update(logprobs=True, top_logprobs=0)
            extra["logprob_token_ids"] = label_token_ids
        return self.client.chat.completions.create(
            model=self.model, messages=messages, max_completion_tokens=max_tokens,
            extra_body=extra or None, **options,
        )
