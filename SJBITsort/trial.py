import gradio as gr
import torch
from gradio_pdf import PDF
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
import PyPDF2  # Import for PDF text extraction

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.float16)
model = model.to('cuda:0')

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as pdf_reader:
        pdf_reader = PyPDF2.PdfReader(pdf_reader)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def predict(message, history):

    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    messages = "".join(["".join(["\n<human>:"+item[0], "\n<bot>:"+item[1]])  #curr_system_message +
                for item in history_transformer_format])

    model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=1.0,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
        )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message  = ""
    for new_token in streamer:
        if new_token != '<':
            partial_message += new_token
            yield partial_message

def handle_chat_and_pdf(message, pdf_file):  # Removed `history` argument
    if pdf_file:
        pdf_text = extract_text_from_pdf(pdf_file)
        message = f"{message}\nPDF content:\n{pdf_text}"

    response = predict(message, [])  # Provide an empty history list
    return response

with gr.Blocks() as demo:
    pdf = PDF(label="Upload a PDF", interactive=True)  # Ensure `gradio_pdf` is installed
    chatbot = gr.ChatInterface(handle_chat_and_pdf)  # Removed `history` and `live` arguments

    pdf.upload(handle_chat_and_pdf, inputs=[chatbot], outputs=chatbot)

demo.launch()
