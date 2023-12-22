
# Automated Resume Shortlisting Assistant 



<br>

This repository contains our attempt to create an Automated Resume Shortlisting Assistant using Hugging Face Transformers.

<br>

## Screenshots


<br>

## Prerequisites

Python 3.12
PyTorch
Gradio
Transformers library
<br>

## Installation:

Install the required libraries using the following command:

`
pip install torch gradio transformers
`
<br>

Usage:

Open your web browser and go to http://localhost:7860 to interact with the chat interface.
<br>

Customizing the Model depending on your hardware capablities
You can customize the model by modifying the following lines in the chat_app.py file:

<br>

`tokenizer = AutoTokenizer.from_pretrained("your-model-name")
model = AutoModelForCausalLM.from_pretrained("your-model-name", torch_dtype=torch.float16)`


<br>


<br>

## Additional Configuration
You can adjust the following parameters in the chat_app.py file to fine-tune the model's behavior:
<br>
max_new_tokens: Maximum number of tokens to generate in each response.
do_sample: Whether to use sampling for generating responses.
top_p: The nucleus sampling parameter.
top_k: The top-k sampling parameter.
temperature: The temperature for sampling.
num_beams: The number of beams for beam search.
stopping_criteria: Criteria for stopping the generation.

<br>

## Acknowledgements
SJBIT Code Fiesta
Gradio: https://gradio.app/
Hugging Face Transformers: https://huggingface.co/transformers/

<br>

## License
This project is licensed under the GNU License - see the LICENSE file for details.