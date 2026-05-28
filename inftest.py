import os
from typing import List, Tuple

import gradio as gr
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

MODEL_ID = "Balab2021/qwen-workflow-planner-qwen2p5-lora"
HF_TOKEN_KEYS = os.getenv("HF_TOKEN_KEYS")


def get_hf_token() -> str:
	if not HF_TOKEN_KEYS:
		raise RuntimeError(
			"Missing HF_TOKEN_KEYS environment variable. "
			"Set it to one or more token env var names (comma-separated), "
			"for example: HF_TOKEN_KEYS=HF_TOKEN"
		)

	raw_value = HF_TOKEN_KEYS.strip().strip("\"'")

	# Allow HF_TOKEN_KEYS to hold a direct Hugging Face token.
	if raw_value.startswith("hf_"):
		return raw_value

	keys = [key.strip() for key in raw_value.split(",") if key.strip()]

	if not keys:
		raise RuntimeError(
			"HF_TOKEN_KEYS is empty. "
			"Set it to one or more token env var names, for example: HF_TOKEN"
		)

	for key in keys:
		token = os.getenv(key)
		if token:
			return token.strip().strip("\"'")
	raise RuntimeError(
		"Missing Hugging Face token. None of the env vars listed in "
		f"HF_TOKEN_KEYS contain a token value. Checked keys: {', '.join(keys)}"
	)


def build_messages(history: List[Tuple[str, str]], user_message: str):
	messages = []
	for user_text, assistant_text in history:
		if user_text:
			messages.append({"role": "user", "content": user_text})
		if assistant_text:
			messages.append({"role": "assistant", "content": assistant_text})
	messages.append({"role": "user", "content": user_message})
	return messages


def create_app():
	load_dotenv()
	token = get_hf_token()

	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
	model = AutoModelForCausalLM.from_pretrained(
		MODEL_ID,
		token=token,
		torch_dtype="auto",
		device_map="auto",
	)

	def chat_fn(
		message: str,
		history: List[Tuple[str, str]],
		temperature: float,
		max_new_tokens: int,
	) -> str:
		messages = build_messages(history, message)
		prompt = tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
		)

		inputs = tokenizer(prompt, return_tensors="pt")
		inputs = {k: v.to(model.device) for k, v in inputs.items()}

		with torch.no_grad():
			output_ids = model.generate(
				**inputs,
				max_new_tokens=max_new_tokens,
				temperature=temperature,
				do_sample=temperature > 0,
				pad_token_id=tokenizer.eos_token_id,
			)

		generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
		response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
		return response

	demo = gr.ChatInterface(
		fn=chat_fn,
		additional_inputs=[
			gr.Slider(0.0, 1.5, value=0.2, step=0.05, label="Temperature"),
			gr.Slider(32, 2048, value=512, step=32, label="Max New Tokens"),
		],
		title="Qwen Workflow Planner Chat",
		description=f"Model: {MODEL_ID}",
	)
	return demo


if __name__ == "__main__":
	app = create_app()
	server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
	server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
	app.launch(server_name=server_name, server_port=server_port, inbrowser=True)
