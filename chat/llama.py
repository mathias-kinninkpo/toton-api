from huggingface_hub import InferenceClient
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import os


class LlamaLLM(LLM):
    

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

        # Préparation des messages pour la requête
        messages = [{"role": "user", "content": prompt}]

        client = InferenceClient(model='meta-llama/Meta-Llama-3-8B-Instruct', token=os.getenv("HF_TOKEN"))
        
        # Effectuer la requête à l'API en utilisant le client Inference
        response_content = ""
        try:
            for message in client.chat_completion(
                messages=messages,
                max_tokens=500,
                stream=True
            ):
                delta_content = message.choices[0].delta.get("content", "")
                response_content += delta_content

        except Exception as e:
            print(f"Error during Llama API call: {e}")
            raise

        return response_content

    @property
    def _identifying_params(self) -> dict:
        return {"llmUrl": "meta-llama/Meta-Llama-3-8B-Instruct"}
