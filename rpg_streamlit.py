import os
import streamlit as st
import re
import pandas as pd
from llama_index.core import Settings
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from tenacity import retry, wait_random_exponential, stop_after_attempt
import json
from llama_index.core.base.llms.types import ChatMessage, MessageRole




# set up model, index, embedding, nvidia API -------------------------------------------------------------------------

# Set NVIDIA API Key (Hard-coded)
nvidia_api_key = "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"  # Replace with your actual NVIDIA API key
os.environ["NVIDIA_API_KEY"] = nvidia_api_key
assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
print(f"NVIDIA API key is set: {nvidia_api_key[:5]}...")  # Masking for display

# Initialize the NVIDIA LLM using Settings
Settings.llm = NVIDIA(model="meta/llama-3.1-405b-instruct")
# Initialize the NVIDIA embedding model
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
# Set the text splitter
Settings.text_splitter = SentenceSplitter(chunk_size=400)

# Function to load documents using SimpleDirectoryReader
def load_documents_from_directory(directory_path):
    documents = SimpleDirectoryReader(directory_path).load_data()
    return documents

# Load combined documents
COMBINED_DOCS_DIR = '/home/polabs2/Code/RPG_teacher/data/combined'  # Replace with your actual path
combined_documents = load_documents_from_directory(COMBINED_DOCS_DIR)

# Create the index using VectorStoreIndex
index = VectorStoreIndex.from_documents(combined_documents)







# set up tool calls for each RPG scenario --------------------------------------------------------------------------
function_schemas = [
    {
        "name": "describe_setting",
        "description": "Generates a description of the RPG setting based on the given educational text and novel content.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The educational text or textbook content."},
                "novel": {"type": "string", "description": "The fantasy novel content."}
            },
            "required": ["text", "novel"]
        }
    },
    {
        "name": "describe_question",
        "description": "Creates a question and answer based on the setting description, educational text, and novel content.",
        "parameters": {
            "type": "object",
            "properties": {
                "setting_description": {"type": "string", "description": "Description of the current setting."},
                "text": {"type": "string", "description": "The educational text or textbook content."},
                "novel": {"type": "string", "description": "The fantasy novel content."}
            },
            "required": ["setting_description", "text", "novel"]
        }
    },
    {
        "name": "handle_answer",
        "description": "Evaluates the user's answer and decides whether to give a hint or grade the answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_text": {"type": "string", "description": "The user's response."},
                "grade_level": {"type": "string", "description": "The educational grade level for grading purposes."},
                "question_answer": {"type": "string", "description": "The correct answer to the question."}
            },
            "required": ["user_text", "question_answer"]
        }
    },
    {
        "name": "player_choice",
        "description": "Determines the next step in the story based on the player's choice.",
        "parameters": {
            "type": "object",
            "properties": {
                "previous_choice": {"type": "string", "description": "The player's previous choice or action."}
            },
            "required": ["previous_choice"]
        }
    }
]



def execute_function_call(function_name, arguments):
    if function_name == 'describe_setting':
        text = arguments.get('text', '')
        novel = arguments.get('novel', '')
        # Call the actual function
        result = describe_setting(text, novel)
        # Format the result for display
        formatted_result = result  # `describe_setting` returns a user-friendly string
        return formatted_result
    elif function_name == "describe_question":
        topic = arguments.get('topic', '')
        difficulty = arguments.get('difficulty', 'medium')
        return describe_question(topic, difficulty)
    elif function_name == "handle_answer":
        user_answer = arguments.get('user_answer', '')
        question = arguments.get('question', '')
        return handle_answer(user_answer, question)
    elif function_name == "player_choice":
        choices = arguments.get('choices', [])
        return player_choice(choices)
    else:
        return f"Function '{function_name}' not recognized."


def describe_setting(text, novel):
    """
    Generates a vivid setting description that combines textbook content and novel elements.
    """
    # Construct the prompt
    prompt = f"""
    Using the following textbook content and novel excerpt, create a vivid and immersive setting description for an RPG adventure. Integrate key concepts from the textbook into the world of the novel.

    Textbook Content:
    {text}

    Novel Excerpt:
    {novel}

    Please provide a detailed setting description that blends these elements seamlessly.
    """
    # Call the LLM
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a creative and descriptive storyteller."),
        ChatMessage(role=MessageRole.USER, content=prompt)
    ]
    response = llm_predict_with_retry(messages)
    setting_description = response.message.content.strip()
    return setting_description


def describe_question(topic, difficulty):
    """
    Generates a challenging question based on the given topic and difficulty level.
    """
    # Construct the prompt
    prompt = f"""
    Create a {difficulty}-level question related to the following topic for an educational RPG adventure:

    Topic:
    {topic}

    The question should be engaging and appropriate for the adventure setting.
    """
    # Call the LLM
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are an expert educator crafting questions for students."),
        ChatMessage(role=MessageRole.USER, content=prompt)
    ]
    response = llm_predict_with_retry(messages)
    question = response.message.content.strip()
    return question


def handle_answer(user_answer, question):
    """
    Evaluates the user's answer to a question and provides feedback.
    """
    # Construct the prompt
    prompt = f"""
    Evaluate the following answer to a question and provide feedback:

    Question:
    {question}

    User's Answer:
    {user_answer}

    As an educator, indicate whether the answer is correct or incorrect. Provide the correct answer with a brief explanation if necessary.
    """
    # Call the LLM
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are an expert educator evaluating student's answers."),
        ChatMessage(role=MessageRole.USER, content=prompt)
    ]
    response = llm_predict_with_retry(messages)
    feedback = response.message.content.strip()
    return feedback


def player_choice(choices):
    """
    Presents choices to the player and returns a formatted string for display.
    """
    # Format the choices
    choices_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
    prompt = f"""
    Present the following choices to the player in an RPG adventure:

    {choices_text}

    Encourage the player to make a selection by typing the corresponding number.
    """
    # Since we can't get user input directly here, return the formatted choices
    return prompt
def load_chapter_content(chapter_number):
    # Create a query engine
    query_engine = index.as_query_engine(similarity_top_k=5)

    # Formulate the query to retrieve the specific chapter's combined document
    query_str = f"chapter_{chapter_number}_combined.txt"

    # Retrieve the relevant document
    response = query_engine.query(query_str)

    # Check if the response contains source nodes
    if not response or not response.source_nodes:
        return None

    # Extract the content from the retrieved document
    doc_content = response.source_nodes[0].node.get_content()
    return doc_content

def get_textbook_content(textbook_name, chapter_number):
    # Load and return the content of the selected textbook and chapter
    # Replace with actual content retrieval logic
    if textbook_name == 'Digital Marketing':
        # Here you can retrieve the content from your index or database
        return "Digital marketing involves the promotion of products or services through digital channels..."
    else:
        return "Textbook content not found."

def get_novel_content(novel_name, chapter_number):
    # Load and return the content of the selected novel and chapter
    # Replace with actual content retrieval logic
    if novel_name == 'The Hobbit':
        return "In a hole in the ground there lived a hobbit..."
    else:
        return "Novel content not found."




# Initial story prompt template
initial_story_prompt = """
You are a storytelling assistant who combines fantasy adventures with educational content.

You have access to the following functions:

1. `describe_setting(text, novel)`: Generates a description of the RPG setting based on the given educational text and novel content.
2. `describe_question(setting_description, text, novel)`: Creates a question and answer based on the setting description, educational text, and novel content.
3. `handle_answer(user_text, grade_level, question_answer)`: Evaluates the user's answer and decides whether to give a hint or grade the answer.
4. `player_choice(previous_choice)`: Determines the next step in the story based on the player's choice.

When you need to perform one of these functions, output a JSON object in the following format:

{"name": "function_name", "parameters": {"arg1": "value1", "arg2": "value2"}}

Do not include any additional text in your response.

Always wait for the player's input before proceeding to the next part of the story.

Given the following information:

Fantasy Chapter Name: {fantasy_chapter_name}

Fantasy Chapter Summary: {fantasy_summary}

Textbook Chapter Name: {textbook_chapter_name}

Textbook Chapter Summary: {textbook_summary}

Sample Questions: {sample_questions}

Create a fun and immersive RPG setting that blends the fantasy narrative with the educational topics. Introduce challenges or puzzles based on the sample questions that the player must solve interactively.

The grade level is {st.session_state.grade_level}.

Please start by calling the `describe_setting` function with the provided textbook and novel content.
"""

# Continuation prompt template
continuation_prompt_template = """
The player responded: "{user_input}"

Continue the story based on the player's response, presenting the next challenge or progressing the narrative accordingly. Introduce one challenge at a time, and wait for the player's next input.
"""

def parse_function_call(content):
    try:
        function_call = json.loads(content)
        if 'name' in function_call and 'parameters' in function_call:
            return function_call['name'], function_call['parameters']
    except json.JSONDecodeError:
        # Attempt to extract JSON from the content
        json_str_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group()
            try:
                function_call = json.loads(json_str)
                if 'name' in function_call and 'parameters' in function_call:
                    return function_call['name'], function_call['parameters']
            except json.JSONDecodeError:
                pass
    return None, None



def extract_section(text, section_name):
    pattern = rf"\*\*{section_name}:\*\*(.*?)(?=\n\*\*|$)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ''

@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
def llm_predict_with_retry(messages):
    try:
        # Debugging: Confirm all messages are ChatMessage instances
        for idx, msg in enumerate(messages):
            if not isinstance(msg, ChatMessage):
                print(f"Error: Message at index {idx} is not a ChatMessage instance.")
                print(f"Message content: {msg}")
                raise TypeError("All messages must be ChatMessage instances.")

        response = Settings.llm.chat(messages=messages)
        return response
    except Exception as e:
        print(f"Exception: {e}")
        raise e



def generate_next_story_segment(chapter, user_input=None):
    messages = []

    if len(st.session_state.storyline) == 0:
        # Retrieve the content for the selected textbook and novel
        textbook_content = get_textbook_content(st.session_state.selected_textbook, chapter)
        novel_content = get_novel_content(st.session_state.selected_novel, chapter)

        # Format the initial prompt
        formatted_prompt = f"""
        You are a storytelling assistant who combines fantasy adventures with educational content.

        You have access to the following functions:

        1. `describe_setting(text, novel)`: Generates a description of the RPG setting based on the given educational text and novel content.
        2. `describe_question(setting_description, text, novel)`: Creates a question and answer based on the setting description, educational text, and novel content.
        3. `handle_answer(user_text, grade_level, question_answer)`: Evaluates the user's answer.
        4. `player_choice(previous_choice)`: Determines the next step in the story based on the player's choice.

        When you need to perform one of these functions, output a JSON object in the following format:

        ```json
        {{"name": "function_name", "parameters": {{"arg1": "value1", "arg2": "value2"}}}}
        ```

        Do not include any additional text in your response.

        The grade level is {st.session_state.grade_level}.

        Please start by calling the `describe_setting` function with the provided textbook and novel content.
        """

        # Prepare the initial messages
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=formatted_prompt))
        # Provide the textbook and novel content
        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=f"""
                        **Textbook Content:**
                        {textbook_content}

                        **Novel Content:**
                        {novel_content}
                        """,
            )
        )
    else:
        # Map roles from session_state to MessageRole
        role_mapping = {
            'system': MessageRole.SYSTEM,
            'user': MessageRole.USER,
            'assistant': MessageRole.ASSISTANT,
            'function': MessageRole.FUNCTION,
        }

        # Convert previous messages to ChatMessage instances
        for msg in st.session_state.storyline[-5:]:
            role_str = msg['role'].lower()
            role = role_mapping.get(role_str, MessageRole.USER)
            messages.append(ChatMessage(role=role, content=msg['content']))

        # Append user's latest input as a ChatMessage
        if user_input:
            messages.append(ChatMessage(role=MessageRole.USER, content=user_input))

        # Debugging: Print types of messages
    for idx, msg in enumerate(messages):
        print(f"Message {idx} type: {type(msg)}")

        # Call the LLM
    response = llm_predict_with_retry(messages)

    while True:
        assistant_message = response.message
        assistant_content = assistant_message.content.strip()

        # Check if the assistant is making a function call
        if assistant_message.additional_kwargs.get('function_call'):
            function_call = assistant_message.additional_kwargs['function_call']
            function_name = function_call['name']
            function_args = function_call['arguments']

            # Execute the function
            function_response = execute_function_call(function_name, function_args)

            # Append the function response to the messages
            messages.append(ChatMessage(role=MessageRole.FUNCTION, content=function_response))

            # Call the assistant again with the function result
            response = llm_predict_with_retry(messages)
        else:
            # Assistant's regular reply
            st.session_state.storyline.append({'role': 'assistant', 'content': assistant_content})
            break  # Exit the loop

    # Parse function call if any
    function_name, function_args = parse_function_call(assistant_content)

    if function_name and function_args:
        # Execute the function
        function_response = execute_function_call(function_name, function_args)

        # Append the function response to the storyline
        function_result_message = json.dumps(function_response)
        st.session_state.storyline.append({'role': 'function', 'content': function_result_message})

        # Prepare messages for LLM continuation
        messages.append(ChatMessage(role=MessageRole.FUNCTION, content=function_result_message))

        # Prompt the assistant to continue the story
        continuation_prompt = "Please continue the story using the function result."
        messages.append(ChatMessage(role=MessageRole.USER, content=continuation_prompt))

        # Debugging: Print types of messages
        for idx, msg in enumerate(messages):
            print(f"Continuation Message {idx} type: {type(msg)}")

        # Call the LLM again
        response = llm_predict_with_retry(messages)
        assistant_message = response.message
        assistant_content = assistant_message.content.strip()
        st.session_state.storyline.append({'role': 'assistant', 'content': assistant_content})


# Initialize session state variables
if 'storyline' not in st.session_state:
    st.session_state.storyline = []
if 'chapter' not in st.session_state:
    st.session_state.chapter = 1
if 'interaction_count' not in st.session_state:
    st.session_state.interaction_count = 0
if 'grade_level' not in st.session_state:
    st.session_state.grade_level = ''
if 'selected_textbook' not in st.session_state:
    st.session_state.selected_textbook = ''
if 'selected_novel' not in st.session_state:
    st.session_state.selected_novel = ''

# Textbook and novel options
textbook_options = ['Digital Marketing']  # Add more textbooks as needed
novel_options = ['The Hobbit']  # Add more novels as needed

# Ensure session state values are valid
if st.session_state.selected_textbook not in textbook_options:
    st.session_state.selected_textbook = textbook_options[0]

if st.session_state.selected_novel not in novel_options:
    st.session_state.selected_novel = novel_options[0]

st.title("Interactive RPG Adventure")
st.write("Embark on a journey that combines fantasy storytelling with educational challenges!")

# Check if the adventure has started
if not st.session_state.storyline:
    # The adventure has not started yet
    st.write("Please set the grade level, select a textbook and a novel, then click 'Go' to start.")

    # Input for grade level
    st.text_input("Enter the grade level (e.g., 'Grade 5'):", key='grade_level')

    # Dropdown for textbook selection
    st.selectbox("Select the textbook:", textbook_options, key='selected_textbook')

    # Dropdown for novel selection
    st.selectbox("Select the novel:", novel_options, key='selected_novel')

    # Button to start the adventure
    if st.button("Go", key='go_button'):
        if st.session_state.grade_level and st.session_state.selected_textbook and st.session_state.selected_novel:
            # Initialize the adventure
            st.session_state.storyline = []
            st.session_state.interaction_count = 0
            st.session_state.chapter = 1  # Reset to chapter 1 or as needed

            # Generate the first setting description
            generate_next_story_segment(st.session_state.chapter)
            st.rerun()
        else:
            st.error("Please fill in all the fields before starting the adventure.")
else:
    # The adventure has started
    # Display the storyline so far
    for message in st.session_state.storyline:
        role = message['role']
        content = message['content']
        if role == 'assistant':
            st.markdown(f"**Narrator:** {content}")
        elif role == 'user':
            st.markdown(f"**You:** {content}")
        elif role == 'function':
            # Display function outputs in a user-friendly way
            st.markdown(f"**[Function Result]:** {content}")

    # Input box for user response
    user_input = st.text_input("Your response:", key='user_input')

    # Button to submit the response
    if st.button("Submit", key='submit_button') and user_input:
        # Append user's input to the storyline
        st.session_state.storyline.append({'role': 'user', 'content': user_input})
        # Generate the next part of the story
        generate_next_story_segment(st.session_state.chapter, user_input)
        # Clear the input box
        st.rerun()

    # Restart Adventure button
    if st.button("Restart Adventure", key='restart_button'):
        st.session_state.storyline = []
        st.session_state.interaction_count = 0
        st.session_state.chapter = 1
        st.session_state.grade_level = ''
        st.session_state.selected_textbook = ''
        st.session_state.selected_novel = ''
        st.experimental_rerun()