import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI


# Load environment variables
load_dotenv()


# Get configuration settings
azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
azure_oai_key = os.getenv("AZURE_OAI_KEY")
azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")


# Configure the Azure OpenAI client
client = AsyncAzureOpenAI(
    azure_endpoint=azure_oai_endpoint,
    api_key=azure_oai_key,
    api_version="2024-02-15-preview"
)


async def call_openai_model(system_message, user_message, model):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
   
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=800
    )
   
    return response.choices[0].message.content


st.title("Azure OpenAI Chat Interface")


# Text area for system message
system_message = st.text_area("Enter system message:", height=100)


# Text input for user message
user_message = st.text_input("Enter your message:")


# Button to send the message
if st.button("Send"):
    if system_message and user_message:
        with st.spinner("Generating response..."):
            response = asyncio.run(call_openai_model(system_message, user_message, azure_oai_deployment))
        st.text_area("Response:", value=response, height=300)
    else:
        st.warning("Please enter both system message and user message.")


# Display the current environment variables
st.sidebar.header("Current Configuration")
st.sidebar.text(f"Endpoint: {azure_oai_endpoint}")
st.sidebar.text(f"Deployment: {azure_oai_deployment}")
