import requests
from typing import Any, List, Optional
from langchain.llms.base import LLM
import time
from pydantic import Extra
from langchain.callbacks.manager import CallbackManagerForLLMRun


class LlamaLLM(LLM):
    llm_url = 'https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct'
    max_retries = 5  # Maximum de tentatives en cas d'erreur 429

    class Config:
        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        return "Llama3 8B"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        payload = {
            "inputs": prompt,
        }

        headers = {
            "Content-Type": "application/json",
            'Authorization': f'Bearer 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'  # Assurez-vous que le token est correct
        }

        # Boucle pour gérer les tentatives de nouvelle requête en cas de surcharge du serveur
        for attempt in range(self.max_retries):
            response = requests.post(self.llm_url, json=payload, headers=headers, verify=False)

            if response.status_code == 429:  # Trop de requêtes
                print(f"Too Many Requests. Retrying attempt {attempt + 1}/{self.max_retries}...")
                time.sleep(2 ** attempt)  # Exponentiel backoff pour éviter la surcharge
                continue

            try:
                response.raise_for_status()  # Lever une exception si la requête échoue
            except requests.exceptions.HTTPError as e:
                print(f"Request failed: {e}")
                print(f"Response content: {response.content}")  # Afficher le contenu de la réponse pour mieux comprendre l'erreur
                raise

            try:
                # Assurez-vous que la réponse a bien la structure attendue
                print(response.json())  

                return response.json()[0]['generated_text']
            except (KeyError, IndexError) as e:
                print(f"Error parsing the response: {response.json()}")
                raise

        raise Exception("Failed to get a response from the LLaMA API after several attempts.")

    @property
    def _identifying_params(self) -> dict:
        """Get the identifying parameters."""
        return {"llmUrl": self.llm_url}
