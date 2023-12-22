# AI_Resume
 
# Gradio Chat Application with RedPajama-INCITE-Chat-3B

This repository contains our attempt to have an automated Resume shortlisting assistant Hugging Face Transformers.

## Prerequisites

- Python 3.12
- PyTorch
- Gradio
- Transformers library

Install the required libraries using the following command:


pip install torch gradio transformers


Open your web browser and go to http://localhost:7860 to interact with the chat interface.
Customizing the Model
You can customize the model by modifying the AutoTokenizer.from_pretrained and AutoModelForCausalLM.from_pretrained lines in the chat_app.py file. Choose any model from Hugging Face Transformers.


tokenizer = AutoTokenizer.from_pretrained("your-model-name")
model = AutoModelForCausalLM.from_pretrained("your-model-name", torch_dtype=torch.float16)
Additional Configuration
max_new_tokens: Maximum number of tokens to generate in each response.
do_sample: Whether to use sampling for generating responses.
top_p: The nucleus sampling parameter.
top_k: The top-k sampling parameter.
temperature: The temperature for sampling.
num_beams: The number of beams for beam search.
stopping_criteria: Criteria for stopping the generation.


Acknowledgements

Gradio
Hugging Face Transformers
License


This project is licensed under the MIT License - see the LICENSE file for details.