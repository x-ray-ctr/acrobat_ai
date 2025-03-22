import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_id: str, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"  # "cuda" → "auto" にすることで柔軟性あり
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model.to(device), tokenizer

def build_prompt(message: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": message}]

def prepare_input(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def tokenize_input(tokenizer, text: str, device: torch.device) -> dict:
    return tokenizer([text], return_tensors="pt").to(device)

def generate_response_ids(model, input_ids, max_new_tokens: int = 512) -> torch.Tensor:
    return model.generate(
        input_ids.input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True
    )

def trim_prompt_tokens(generated_ids, input_ids):
    return [
        output_ids[len(input_ids):]
        for output_ids, input_ids in zip(generated_ids, input_ids.input_ids)
    ]

def decode_responses(tokenizer, ids) -> list[str]:
    return tokenizer.batch_decode(ids, skip_special_tokens=True)

def run_chat(model_id: str, user_input: str):
    model, tokenizer = load_model_and_tokenizer(model_id, Device)

    messages = build_prompt(user_input)
    prompt_text = prepare_input(tokenizer, messages)
    model_inputs = tokenize_input(tokenizer, prompt_text, Device)
    generated_ids = generate_response_ids(model, model_inputs)

    trimmed_ids = trim_prompt_tokens(generated_ids, model_inputs)
    responses = decode_responses(tokenizer, trimmed_ids)
    
    return responses[0]

if __name__ == "__main__":
    model_id = "Qwen/Qwen2-7B-Instruct"
    prompt = "Give me a short introduction to large language model"
    response = run_chat(model_id, prompt)
    print(response)
