import os
import openai
import tempfile
import gradio as gr

from gtts import gTTS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Number of queries to retrieve
queries_to_retrive = 5
# Global variable to hold the chain
qa_chain = None

def initialize_chroma_from_pdf(pdf_file):
    """
    The function initializes a ChromaDB vector store from a PDF file by loading and splitting the
    document text into manageable chunks and creating embeddings for each chunk.
    
    :return: The function `initialize_chroma_from_pdf` returns a ChromaDB vector store created from the
    documents extracted from the provided PDF file.
    """
    loader = PyPDFLoader(pdf_file.name)
    documents = loader.load()

    # Split the documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    # Create a ChromaDB vector store
    vectorstore = Chroma.from_documents(split_documents, embeddings)
    return vectorstore

def initialize_chain(vectorstore):
    """
    The function creates a retrieval-based question answering system using a language
    model and a vector store retriever.
    
    :return: returns a Question-Answer (QA) chain that consists of an
    OpenAI language model (llm) and a retriever created from the provided vector store.
    """
    llm = OpenAI()

    retriever = vectorstore.as_retriever(search_kwargs={'k': queries_to_retrive})
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    return qa_chain

def upload_and_initialize(pdf_file):
    """
    The function uploads a PDF file and initializes a RAG model for asking
    questions using a microphone.
    
    :param pdf_file: The `pdf_file` parameter is the PDF file that you want to upload and use for
    initializing the RAG (Retrieval-Augmented Generation) model.
    :return:  returns the string with a message.
    """
    global qa_chain
    vectorstore = initialize_chroma_from_pdf(pdf_file)
    
    # Initialize the RAG chain
    qa_chain = initialize_chain(vectorstore)
    return "PDF uploaded and RAG model initialized. You can now ask questions using your microphone."

def speech_to_text(audio_file_path):
    """
    The function takes an audio file path as input, transcribes the audio using
    OpenAI's API, and returns the transcribed text.
    
    :param audio_file_path: The `audio_file_path` parameter is a string that represents the file path to
    the audio file that you want to convert from speech to text. This file should be in a format that
    can be processed for speech recognition, such as WAV or MP3.
    :return: returns the transcribed text.
    """
    audio_file = open(audio_file_path, "rb")
    response = openai.Audio.transcribe("whisper-1", audio_file)
    return response['text']

# def speech_to_text(audio_file_path):
#     client = Client("https://openai-whisper.hf.space/")
#     result = client.predict(
#                     audio_file_path,
#                     "transcribe",	# str in 'Task' Radio component
#                     api_name="/predict"
#             )
#     # print(result)
#     return result

# def text_to_speech(text):
#     client = Client("https://suno-bark.hf.space/")
#     result = client.predict(
#                     text,
#                     fn_index=1
#     )
#     print(result)
#     return result

# Function to process audio input and generate a response
def process_audio_question(audio_file):
    """
    The function takes an audio file as input, converts the speech to text,
    generates a response using a QA chain, and then converts the response text to an audio file using
    gTTS.
    
    :param audio_file: input audio file with the query
    :return: It returns the file path of the generated audio file.
    """
    if qa_chain is None:
        return "Please upload a PDF file first."

    query_text = speech_to_text(audio_file)
    print('Query:', query_text)

    # Generate the response using the RAG chain
    response_text = qa_chain.run(query_text)
    print('Response:', response_text)

    # retrieval_info = response_text["result_source"]
    # print('Retrieval Info:', retrieval_info)
    # response = text_to_speech(response_text)
    # print(response)
    
    # Convert the response to audio using gTTS
    tts = gTTS(text=response_text, lang='en')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts.save(temp_file.name)
    return temp_file.name

with gr.Blocks() as iface:
    gr.Markdown("# Voice enabled RAG Chatbot")
    gr.Markdown("Upload a PDF file to initialize the model, then interact with the chatbot using your microphone.")

    with gr.Row():
        with gr.Column():
            # PDF upload and initialization
            pdf_upload = gr.File(label="Upload a PDF file", file_types=[".pdf"])
            upload_button = gr.Button("Upload and Initialize")
            output_message = gr.Textbox()

            upload_button.click(upload_and_initialize, inputs=pdf_upload, outputs=output_message)

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Ask a question")
            response_audio = gr.Audio(type="filepath", label="Response Audio")
            audio_input.change(process_audio_question, inputs=audio_input, outputs=response_audio)

# Launch the Gradio interface
iface.launch()