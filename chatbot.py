from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import GPT4All 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser

documents = PyPDFLoader('/Users/soodeh/Desktop/LLM/Chatbot/pdf_files/Profile.pdf').load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,chunk_overlap=64)
texts = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
faiss_index = FAISS.from_documents(texts, embeddings)
faiss_index.save_local("/Users/soodeh/Desktop/LLM/Chatbot/index")



from colorama import Fore, Style, init
init(autoreset=True)



embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# load vector store
print("loading indexes")
faiss_index = FAISS.load_local("/Users/soodeh/Desktop/LLM/Chatbot/index", embeddings,  allow_dangerous_deserialization=True )
print("index loaded")
gpt4all_path = "/Users/soodeh/.cache/gpt4all/Meta-Llama-3-8B-Instruct.Q4_0.gguf"

# # Set your query here manually

callback_manager = BaseCallbackManager([StreamingStdOutCallbackHandler()])
llm = GPT4All(model=gpt4all_path, verbose=True, allow_download=True, max_tokens=200, temp=0.5)
print(Fore.BLACK +"")
print(Fore.BLACK +"")
question = input(Fore.GREEN +"  Ask me anything: ")

while question!="end":
    matched_docs = faiss_index.similarity_search(question, 4)
    context = ""
    for doc in matched_docs:
        context = context + doc.page_content + " \n\n "

    template = """
    If the context is not relevant, please answer the question by using your own knowledge about the topic
    Context: {context}
    - -
    Question: {question}
    Answer: Let's think step by step.  No Next Question.""" 


    prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
    chain = prompt | llm | StrOutputParser()

    print(Fore.BLACK +"")
    print(Fore.WHITE +chain.invoke(question))
    print()
    question = input(Fore.GREEN +"  Ask me anything: ")
    print(Fore.BLACK +"")
