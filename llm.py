from litellm import completion
from litellm.files.main import ModelResponse
from dataclasses import dataclass

import re


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    total_tokens: int = 0

    def __post_init__(self):
        self.total_tokens = self.input_tokens + self.output_tokens


def extract_tag(text: str, tag="blog") -> str | None:
    matches = re.findall(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if matches:
        return matches[0].strip()


def query_llm(
    prompt: str, model: str = "gemini/gemini-2.5-flash-preview-04-17"
) -> LLMResponse:
    messages = [{"content": prompt, "role": "user"}]
    response = completion(model=model, messages=messages)

    if not isinstance(response, ModelResponse):
        raise ValueError(f"Expected ModelResponse, got {type(response)}")

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    match response.choices:
        case [choice]:
            message = choice.message.to_dict()
            if message.get("role") == "assistant" and "content" in message:
                return LLMResponse(
                    content=message["content"],
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                )
            raise ValueError(f"Unexpected message format: {message}")
        case [choice, *_]:
            # Use first choice if multiple are returned
            message = choice.message.to_dict()
            if message.get("role") == "assistant" and "content" in message:
                return LLMResponse(
                    content=message["content"],
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                )
            raise ValueError(f"Unexpected message format in first choice: {message}")
        case _:
            raise ValueError(f"Unexpected response format: {response}")


if __name__ == "__main__":
    response = query_llm("What is the capital of France?")
    print(f"Content: {response.content}")
    print(f"Input tokens: {response.input_tokens}")
    print(f"Output tokens: {response.output_tokens}")
    print(f"Total tokens: {response.total_tokens}")
    print(f"Model: {response.model}")
