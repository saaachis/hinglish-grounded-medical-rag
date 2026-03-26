"""
Grounded Generator.

Generates Hinglish clinical explanations conditioned on retrieved
evidence, ensuring factual grounding and minimizing hallucinations.

Uses Llama-3.1-8B-Instant via Groq API for inference.
"""

from __future__ import annotations

import logging
import time

from groq import Groq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_GROUNDED = (
    "You are a medical assistant helping patients understand their symptoms.\n"
    "You MUST base your response strictly on the clinical evidence provided below.\n"
    "Respond in Hinglish (mix of Hindi and English) since the patient communicates in Hinglish.\n"
    "Keep the response concise (3-5 sentences). Only state facts supported by the evidence.\n"
    "If the evidence does not cover something, say you cannot confirm it."
)

SYSTEM_PROMPT_ZERO_SHOT = (
    "You are a medical assistant helping patients understand their symptoms.\n"
    "Respond in Hinglish (mix of Hindi and English) since the patient communicates in Hinglish.\n"
    "Keep the response concise (3-5 sentences).\n"
    "You do NOT have access to any clinical reports or test results."
)

MAX_EVIDENCE_WORDS = 400


class GroundedGenerator:
    """Evidence-grounded Hinglish explanation generator.

    Produces Hinglish clinical explanations conditioned on retrieved
    medical evidence via the Groq cloud API.

    Parameters
    ----------
    api_key : str
        Groq API key.
    model_name : str
        Model identifier for Groq.
    max_tokens : int
        Maximum output tokens per generation.
    temperature : float
        Sampling temperature.
    max_retries : int
        Number of retries on API failure.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "llama-3.1-8b-instant",
        max_tokens: int = 300,
        temperature: float = 0.3,
        max_retries: int = 3,
    ):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

    def _call_llm(self, system: str, user_msg: str) -> str:
        """Call Groq API with retry logic."""
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning("Groq API error (attempt %d): %s", attempt + 1, e)
                if attempt < self.max_retries - 1:
                    time.sleep(3 * (attempt + 1))
        return "[API_ERROR] Could not generate response."

    def build_grounded_prompt(
        self,
        query: str,
        evidence_texts: list[str],
    ) -> str:
        """Build a user prompt that injects retrieved evidence for grounding.

        Parameters
        ----------
        query : str
            Hinglish patient query.
        evidence_texts : list[str]
            Retrieved evidence passages to inject.

        Returns
        -------
        str
            Formatted prompt with evidence context.
        """
        combined = "\n\n---\n\n".join(
            " ".join(t.split()[:MAX_EVIDENCE_WORDS]) for t in evidence_texts
        )
        return (
            f"Clinical Evidence:\n{combined}\n\n"
            f"Patient Query:\n{query}\n\n"
            f"Respond based strictly on the clinical evidence above."
        )

    def generate(
        self,
        query: str,
        evidence_texts: list[str],
    ) -> str:
        """Generate a grounded Hinglish explanation.

        Parameters
        ----------
        query : str
            Hinglish patient query.
        evidence_texts : list[str]
            Retrieved clinical report texts.

        Returns
        -------
        str
            Generated Hinglish clinical explanation.
        """
        user_msg = self.build_grounded_prompt(query, evidence_texts)
        return self._call_llm(SYSTEM_PROMPT_GROUNDED, user_msg)

    def generate_zero_shot(self, query: str) -> str:
        """Generate a zero-shot response without evidence.

        Parameters
        ----------
        query : str
            Hinglish patient query.

        Returns
        -------
        str
            Generated Hinglish response based on general knowledge only.
        """
        user_msg = (
            f"Patient Query:\n{query}\n\n"
            f"Respond based on your general medical knowledge only."
        )
        return self._call_llm(SYSTEM_PROMPT_ZERO_SHOT, user_msg)
