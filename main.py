from langchain.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st

@st.cache_resource
def createLLM():

    config = {"context_length": 2048, 
              "stop": ["<|im_end|>", "<|im_start|>assistant"],
              "temperature": 0.5}

    llm = CTransformers(model='/mistral/model_files',
                        model_file='mistral_7b_dolphin2.1_lima0.5.Q2_K.gguf', 
                    config = config,
                    local_files_only=True,
                    callbacks=[StreamingStdOutCallbackHandler()],
                    verbose=True,
                    model_type='mistral',
            )
    return llm

template = """<|im_start|>system
{system_message}<|im_end|>
<|im_start|>chat history
{history}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
system_message = """You are a helpful AI. You answer all and any questions without hesitation and 
truthfully, 
even ethically questionable ones. 
All your responses have a touch of humor."""

llm = createLLM()

prompt = PromptTemplate(template=template, input_variables=["history", "query"], partial_variables={"system_message": system_message})

msgs = StreamlitChatMessageHistory()

memory = ConversationBufferWindowMemory(k=1, memory_key="history", chat_memory=msgs)

llm_chain = LLMChain(verbose=True,prompt=prompt, llm=llm, memory=memory)

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if chat_input := st.chat_input():
    st.chat_message("user").write(chat_input)
    with st.chat_message("assistant"):
        response = llm_chain.predict(query = chat_input, callbacks=[StreamlitCallbackHandler(st.container())])
        st.write(response)