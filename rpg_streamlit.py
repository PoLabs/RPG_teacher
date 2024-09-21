import os
import streamlit as st
import re
import pandas as pd
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    PromptTemplate,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from tenacity import retry, wait_random_exponential, stop_after_attempt

from llama_index import ServiceContext



# Set NVIDIA API Key
if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    nvidia_api_key = st.sidebar.text_input("Enter your NVIDIA API key:", type="password")
    if nvidia_api_key:
        assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
        os.environ["NVIDIA_API_KEY"] = nvidia_api_key
else:
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY")


from llama_index.llms.nvidia import NVIDIALLM
from llama_index.embeddings.nvidia import NVIDIAEmbedding

# Initialize the LLM
llm = NVIDIALLM(model="meta/llama-3.1-405b-instruct")

# Initialize the embedding model
embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

# Initialize the embedding model
#Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
# Set up the text splitter
#Settings.text_splitter = SentenceSplitter(chunk_size=400)

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# Load combined documents
COMBINED_DOCS_DIR = '/home/polabs2/Code/RPG_teacher/data/combined'  # Replace with your actual path
combined_documents = SimpleDirectoryReader(COMBINED_DOCS_DIR).load_data()
# Create the index
#index = VectorStoreIndex.from_documents(combined_documents)
index = VectorStoreIndex.from_documents(combined_documents, service_context=service_context)


# Initial story prompt template
initial_story_prompt = PromptTemplate(template="""
You are a storyteller who combines fantasy adventures with educational content.

Given the following information:

**Fantasy Chapter Name:**
{fantasy_chapter_name}

**Fantasy Chapter Summary:**
{fantasy_summary}

**Textbook Chapter Name:**
{textbook_chapter_name}

**Textbook Chapter Summary:**
{textbook_summary}

**Sample Questions:**
{sample_questions}

Create a fun and immersive RPG setting that blends the fantasy narrative with the educational topics. Introduce challenges or puzzles based on the sample questions that the player must solve interactively. Present one challenge at a time, and wait for the player's input before proceeding to the next part of the story.

Please start the story now.
""")

# Continuation prompt template
continuation_prompt_template = PromptTemplate(template="""
The player responded: "{user_input}"

Continue the story based on the player's response, presenting the next challenge or progressing the narrative accordingly. Introduce one challenge at a time, and wait for the player's next input.
""")

def extract_section(text, section_name):
    pattern = rf"\*\*{section_name}:\*\*(.*?)(?=\n\*\*|$)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ''

@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
def llm_predict_with_retry(prompt_template, **kwargs):
    return Settings.llm.predict(prompt=prompt_template, **kwargs)

def generate_next_story_segment(chapter, user_input=None):
    # Create a query engine
    query_engine = index.as_query_engine(similarity_top_k=1)

    # Formulate the query to retrieve the specific chapter's combined document
    query_str = f"chapter_{chapter}_combined.txt"

    # Retrieve the relevant document
    response = query_engine.query(query_str)

    # Check if the response contains source nodes
    if not response or not response.source_nodes:
        st.error(f"No document found for chapter {chapter}")
        return

    # Extract the content from the retrieved document
    doc_content = response.source_nodes[0].node.get_content()

    # Extract sections using the helper function
    fantasy_chapter_name = extract_section(doc_content, 'Fantasy Chapter Name')
    fantasy_summary = extract_section(doc_content, 'Fantasy Chapter Summary')
    textbook_chapter_name = extract_section(doc_content, 'Textbook Chapter Name')
    textbook_summary = extract_section(doc_content, 'Textbook Chapter Summary')
    sample_questions = extract_section(doc_content, 'Sample Questions')

    if st.session_state.interaction_count == 0:
        # Generate the initial story segment
        initial_story = llm_predict_with_retry(
            prompt_template=initial_story_prompt,
            fantasy_chapter_name=fantasy_chapter_name,
            fantasy_summary=fantasy_summary,
            textbook_chapter_name=textbook_chapter_name,
            textbook_summary=textbook_summary,
            sample_questions=sample_questions
        )
        st.session_state.storyline.append({'role': 'assistant', 'content': initial_story})
    else:
        # Generate the continuation based on user input
        next_segment = llm_predict_with_retry(
            prompt_template=continuation_prompt_template,
            user_input=user_input
        )
        st.session_state.storyline.append({'role': 'assistant', 'content': next_segment})

    st.session_state.interaction_count += 1


if 'storyline' not in st.session_state:
    st.session_state.storyline = []
if 'chapter' not in st.session_state:
    st.session_state.chapter = 1  # You can allow the user to select the chapter
if 'interaction_count' not in st.session_state:
    st.session_state.interaction_count = 0

st.title("Interactive RPG Adventure")
st.write("Embark on a journey that combines fantasy storytelling with educational challenges!")

# Display the storyline so far
for message in st.session_state.storyline:
    if message['role'] == 'assistant':
        st.markdown(f"**Narrator:** {message['content']}")
    else:
        st.markdown(f"**You:** {message['content']}")

# Input box for user response
if st.session_state.interaction_count == 0 or st.session_state.storyline[-1]['role'] == 'assistant':
    user_input = st.text_input("Your response:", key='user_input')

    # Button to submit the response
    if st.button("Submit") and user_input:
        # Append user's input to the storyline
        st.session_state.storyline.append({'role': 'user', 'content': user_input})
        # Generate the next part of the story
        generate_next_story_segment(st.session_state.chapter, user_input)
        # Clear the input box
        st.experimental_rerun()
else:
    st.write("Awaiting the narrator's response...")

if st.button("Restart Adventure"):
    st.session_state.storyline = []
    st.session_state.interaction_count = 0
    st.experimental_rerun()

MAX_MESSAGES = 10  # Adjust based on your model's capabilities
if len(st.session_state.storyline) > MAX_MESSAGES:
    st.session_state.storyline = st.session_state.storyline[-MAX_MESSAGES:]
