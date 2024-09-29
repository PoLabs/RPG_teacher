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
TOP_K = 5  # Global constant

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

'''# Function to load documents using SimpleDirectoryReader
def load_documents_from_directory(directory_path):
    documents = SimpleDirectoryReader(directory_path).load_data()
    return documents

# Load combined documents
COMBINED_DOCS_DIR = '/home/polabs2/Code/RPG_teacher/data/combined'  # Replace with your actual path
combined_documents = load_documents_from_directory(COMBINED_DOCS_DIR)

# Create the index using VectorStoreIndex
index = VectorStoreIndex.from_documents(combined_documents)'''



def load_textbook_documents(textbook_name):
    directory_path = f"/home/polabs2/Code/RPG_teacher/data/textbooks/{textbook_name}"
    documents = SimpleDirectoryReader(directory_path).load_data()
    return documents

def load_novel_documents(novel_name):
    directory_path = f"/home/polabs2/Code/RPG_teacher/data/novels/{novel_name}"
    documents = SimpleDirectoryReader(directory_path).load_data()
    return documents

def create_textbook_index(textbook_name):
    documents = load_textbook_documents(textbook_name)
    textbook_index = VectorStoreIndex.from_documents(documents)
    return textbook_index

def create_novel_index(novel_name):
    documents = load_novel_documents(novel_name)
    novel_index = VectorStoreIndex.from_documents(documents)
    return novel_index




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
    Uses top_k relevant documents from both indexes as context.
    """
    print("\n--- describe_setting called ---")
    print(f"text: {text[:100]}...")
    print(f"novel: {novel[:100]}...")

    # Access indexes from st.session_state
    textbook_index = st.session_state.textbook_index
    novel_index = st.session_state.novel_index
    top_k = 5  # Default value

    # Create query engines for both indexes
    textbook_query_engine = textbook_index.as_query_engine(similarity_top_k=top_k)
    novel_query_engine = novel_index.as_query_engine(similarity_top_k=top_k)

    # Query both indexes
    textbook_response = textbook_query_engine.query(text)
    novel_response = novel_query_engine.query(novel)

    # Extract content from the retrieved documents
    textbook_context_docs = [node.node.get_content() for node in textbook_response.source_nodes]
    novel_context_docs = [node.node.get_content() for node in novel_response.source_nodes]

    # Combine the context documents
    context = "\n".join(textbook_context_docs + novel_context_docs)

    # Construct the prompt
    prompt = f"""
    Using the following context from the textbook and novel, create a vivid and immersive setting description for an RPG adventure. Integrate key concepts from the textbook into the world of the novel.

    Context:
    {context}

    Please provide a detailed setting description that blends these elements seamlessly.
    """
    # Call the LLM
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a creative and descriptive storyteller."),
        ChatMessage(role=MessageRole.USER, content=prompt)
    ]
    response = llm_predict_with_retry(messages)
    setting_description = response.message.content.strip()

    print(f"Generated setting description: {setting_description[:100]}...")
    print("--- describe_setting ended ---\n")

    return setting_description


def describe_question(setting_description, text, novel):
    """
    Generates a challenging question based on the setting description, text, and novel content.
    Includes relevant sample questions from chapter_summary_notes.csv.
    """
    print("\n--- describe_question called ---")
    print(f"Setting description: {setting_description[:100]}...")

    # Access indexes from st.session_state
    textbook_index = st.session_state.textbook_index
    novel_index = st.session_state.novel_index
    top_k = 5  # Default value

    # Retrieve relevant sample questions from chapter_summary_notes.csv
    chapter = st.session_state.chapter
    df = st.session_state.chapter_summary_notes
    chapter_notes = df[df['chapter'] == chapter]

    if not chapter_notes.empty:
        sample_questions = chapter_notes['sample_questions'].values[0]
    else:
        sample_questions = ""

    # Create query engines for both indexes
    textbook_query_engine = textbook_index.as_query_engine(similarity_top_k=top_k)
    novel_query_engine = novel_index.as_query_engine(similarity_top_k=top_k)

    # Query both indexes
    textbook_response = textbook_query_engine.query(text)
    novel_response = novel_query_engine.query(novel)

    # Extract content from the retrieved documents
    textbook_context_docs = [node.node.get_content() for node in textbook_response.source_nodes]
    novel_context_docs = [node.node.get_content() for node in novel_response.source_nodes]

    # Combine the context documents
    context = "\n".join(textbook_context_docs + novel_context_docs)

    # Construct the prompt
    grade_level = st.session_state.grade_level

    prompt = f"""
    Using the following setting description, context, and sample questions, create an engaging question for the player that ties into the RPG adventure. The question should be appropriate for grade level {grade_level} and integrate educational content from the textbook. Please provide the question and its answer.

    Setting Description:
    {setting_description}

    Context:
    {context}

    Sample Questions:
    {sample_questions}

    Please format your response as:

    Question:
    [Your question here]

    Answer:
    [The correct answer here]
    """
    # Call the LLM
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are an expert educator crafting questions for students."),
        ChatMessage(role=MessageRole.USER, content=prompt)
    ]
    response = llm_predict_with_retry(messages)
    question_and_answer = response.message.content.strip()

    print(f"Generated question and answer: {question_and_answer}")
    print("--- describe_question ended ---\n")

    # Parse the response to separate question and answer
    match = re.search(r'Question:\s*(.*?)\s*Answer:\s*(.*)', question_and_answer, re.DOTALL)
    if match:
        question = match.group(1).strip()
        answer = match.group(2).strip()
    else:
        print("Failed to parse question and answer.")
        question = question_and_answer
        answer = ""

    return question, answer

def give_hint(question):
    """
    Provides a hint to the user using RAG results from the textbook index.
    """
    print("\n--- give_hint called ---")
    print(f"Question: {question}")

    # Access textbook index from st.session_state
    textbook_index = st.session_state.textbook_index
    top_k = 5  # Default value

    # Create a query engine for the textbook index
    query_engine = textbook_index.as_query_engine(similarity_top_k=top_k)

    # Retrieve relevant documents
    response = query_engine.query(question)

    # Extract content from the retrieved documents
    context_docs = [node.node.get_content() for node in response.source_nodes]

    # Combine the context documents
    context = "\n".join(context_docs)

    # Construct the prompt
    prompt = f"""
    Using the following context, provide a hint to help the student answer the question. Do not provide the answer directly.

    Question:
    {question}

    Context:
    {context}

    Please provide a helpful hint.
    """
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are an expert educator providing hints to students."),
        ChatMessage(role=MessageRole.USER, content=prompt)
    ]
    response = llm_predict_with_retry(messages)
    hint = response.message.content.strip()

    print(f"Hint: {hint}")
    print("--- give_hint ended ---\n")

    return hint



def handle_answer(user_input, question, question_answer):
    """
    Evaluates the user's input and decides whether it's an answer, a hint request, or something else.
    """
    print("\n--- handle_answer called ---")
    print(f"user_input: {user_input}")
    print(f"question: {question}")

    # Use LLM to analyze the user's input
    prompt = f"""
    Determine if the following user input is an answer to the question, a request for a hint, or something else.

    Question:
    {question}

    User Input:
    {user_input}

    Please respond with one of the following options: 'answer', 'hint_request', or 'other'.
    """
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are an assistant that classifies user input."),
        ChatMessage(role=MessageRole.USER, content=prompt)
    ]
    response = llm_predict_with_retry(messages)
    classification = response.message.content.strip().lower()

    print(f"Classification: {classification}")

    if 'answer' in classification:
        # User provided an answer, grade it
        feedback = grade_answer(user_input, st.session_state.current_question, st.session_state.current_question_answer)
    elif 'hint' in classification:
        # User requested a hint
        feedback = give_hint(question)
    else:
        feedback = "I'm sorry, I didn't understand your response. Could you please try again?"

    print(f"Feedback: {feedback}")
    print("--- handle_answer ended ---\n")

    return feedback


def grade_answer(user_answer, question, correct_answer):
    """
    Grades the user's answer and provides feedback.
    """
    print("\n--- grade_answer called ---")
    print(f"user_answer: {user_answer}")
    print(f"question: {question}")
    print(f"correct_answer: {correct_answer}")

    # Access grade level from st.session_state
    grade_level = st.session_state.grade_level

    # Construct the prompt
    prompt = f"""
    As an educator for grade {grade_level}, grade the following student's answer to the question. Provide a grade from A to F, an explanation, and any necessary references to explain the answer.

    Question:
    {question}

    Correct Answer:
    {correct_answer}

    Student's Answer:
    {user_answer}

    Please provide the grade, explanation, and references.
    """
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are an expert educator grading a student's answer."),
        ChatMessage(role=MessageRole.USER, content=prompt)
    ]
    response = llm_predict_with_retry(messages)
    grading_feedback = response.message.content.strip()

    print(f"Grading Feedback: {grading_feedback}")
    print("--- grade_answer ended ---\n")

    return grading_feedback



def player_choice(choices):
    """
    Presents choices to the player and returns a formatted string for display.
    """
    print("\n--- player_choice called ---")
    print(f"choices: {choices}")

    # Format the choices
    choices_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
    prompt = f"""
    Present the following choices to the player in an RPG adventure:

    {choices_text}

    Encourage the player to make a selection by typing the corresponding number.
    """

    print(f"Choices prompt: {prompt}")
    print("--- player_choice ended ---\n")

    return prompt



def extract_topic_from_textbook(text):
    """
    Extracts a topic from the textbook content.
    For simplicity, we'll return a placeholder or the first sentence.
    """
    # You can implement a more sophisticated method or use NLP techniques
    sentences = text.split('.')
    if sentences:
        return sentences[0]  # Return the first sentence as the topic
    else:
        return "General Topic"


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


def generate_next_story_segment(user_input=None):
    print("\n--- generate_next_story_segment called ---")
    print(f"user_input: {user_input}")

    # Initialize game state if not already done
    if 'game_stage' not in st.session_state:
        st.session_state.game_stage = 'start'
        st.session_state.current_question = ''
        st.session_state.current_question_answer = ''
        st.session_state.current_setting = ''
        st.session_state.current_choices = []
        st.session_state.chapter = 1

    if 'storyline' not in st.session_state:
        st.session_state.storyline = []

    if st.session_state.game_stage == 'start':
        print("Game stage: start")

        # Load the selected textbook and novel content
        text = st.session_state.selected_textbook_content
        novel = st.session_state.selected_novel_content

        # Describe the initial setting
        setting_description = describe_setting(text, novel)
        st.session_state.current_setting = setting_description
        st.session_state.storyline.append({'role': 'assistant', 'content': setting_description})

        # Generate the first question
        question, question_answer = describe_question(setting_description, text, novel)
        st.session_state.current_question = question
        st.session_state.current_question_answer = question_answer
        st.session_state.storyline.append({'role': 'assistant', 'content': question})

        # Update game stage
        st.session_state.game_stage = 'awaiting_answer'


    elif st.session_state.game_stage == 'awaiting_answer':
        print("Game stage: awaiting_answer")

        if user_input is None:
            print("No user input received.")
            return

        # Handle user's answer
        user_answer = user_input
        st.session_state.storyline.append({'role': 'user', 'content': user_answer})
        feedback = handle_answer(user_answer, st.session_state.current_question, st.session_state.current_question_answer)
        st.session_state.storyline.append({'role': 'assistant', 'content': feedback})
        # Check if feedback contains a grade (indicates answer was graded)
        if "Grade:" in feedback or "grade:" in feedback.lower():
            # Present choices to the player
            choices = ["1. Continue your journey", "2. Review the material", "3. Exit the game"]
            st.session_state.current_choices = choices
            choices_prompt = player_choice(choices)
            st.session_state.storyline.append({'role': 'assistant', 'content': choices_prompt})
            # Update game stage
            st.session_state.game_stage = 'awaiting_choice'
        else:
            # If the user asked for a hint or input was not understood, stay in the same stage
            st.session_state.game_stage = 'awaiting_answer'
            elif st.session_state.game_stage == 'awaiting_choice':
            print("Game stage: awaiting_choice")

            if user_input is None:
                print("No user input received.")
                return

            # Handle player's choice
            player_choice_input = user_input
            st.session_state.storyline.append({'role': 'user', 'content': player_choice_input})

            # Process the player's choice
            try:
                choice_index = int(player_choice_input.strip()) - 1
                if choice_index < 0 or choice_index >= len(st.session_state.current_choices):
                    raise ValueError("Invalid choice number.")
                chosen_option = st.session_state.current_choices[choice_index]
                print(f"Player chose: {chosen_option}")
            except ValueError:
                st.session_state.storyline.append(
                    {'role': 'assistant', 'content': "Invalid choice. Please enter a valid number."})
                # Stay in the same stage to prompt for valid input
                return

            # Handle the choice
            if choice_index == 0:
                # Continue the journey: Generate new setting and question
                text = st.session_state.selected_textbook_content
                novel = st.session_state.selected_novel_content
                setting_description = describe_setting(text, novel)
                st.session_state.current_setting = setting_description
                st.session_state.storyline.append({'role': 'assistant', 'content': setting_description})

                question, question_answer = describe_question(setting_description, text, novel)
                st.session_state.current_question = question
                st.session_state.current_question_answer = question_answer
                st.session_state.storyline.append({'role': 'assistant', 'content': question})

                # Update game stage
                st.session_state.game_stage = 'awaiting_answer'
            elif choice_index == 1:
                # Review the material: Provide a summary or hint
                hint = give_hint(st.session_state.current_question)
                st.session_state.storyline.append({'role': 'assistant', 'content': hint})
                # Stay in the same stage
                st.session_state.game_stage = 'awaiting_answer'
            elif choice_index == 2:
                # Exit the game
                st.session_state.storyline.append(
                    {'role': 'assistant', 'content': "Thank you for playing! See you next time."})
                st.session_state.game_stage = 'end'
            else:
                # Invalid choice
                st.session_state.storyline.append({'role': 'assistant', 'content': "Invalid choice. Please try again."})
                # Stay in the same stage

    else:
        st.error("Unknown game stage.")

    print("--- generate_next_story_segment ended ---\n")


# Initialize session state variables
if 'storyline' not in st.session_state:
    st.session_state.storyline = []
if 'game_stage' not in st.session_state:
    st.session_state.game_stage = 'start'
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
if 'textbook_index' not in st.session_state:
    st.session_state.textbook_index = None
if 'novel_index' not in st.session_state:
    st.session_state.novel_index = None
if 'chapter_summary_notes' not in st.session_state:
    st.session_state.chapter_summary_notes = pd.read_csv('data/chapter_summary_notes.csv')
# Initialize other variables as needed


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

# Initialize session state variables (as shown earlier)

# Input for grade level
if st.session_state.game_stage == 'start' and not st.session_state.storyline:
    st.text_input("Enter the grade level (e.g., 'Grade 5'):", key='grade_level')

    # Dropdown for textbook selection
    st.selectbox("Select the textbook:", textbook_options, key='selected_textbook')

    # Dropdown for novel selection
    st.selectbox("Select the novel:", novel_options, key='selected_novel')

    # Button to start the adventure
    # After the user selects the textbook and novel and clicks "Start Adventure"
    if st.button("Start Adventure", key='start_button'):
        if st.session_state.grade_level and st.session_state.selected_textbook and st.session_state.selected_novel:
            # Load the selected textbook and novel content
            st.session_state.selected_textbook_content = get_textbook_content(
                st.session_state.selected_textbook, st.session_state.chapter)
            st.session_state.selected_novel_content = get_novel_content(
                st.session_state.selected_novel, st.session_state.chapter)

            # Create indexes for the selected textbook and novel
            st.session_state.textbook_index = create_textbook_index(st.session_state.selected_textbook)
            st.session_state.novel_index = create_novel_index(st.session_state.selected_novel)

            generate_next_story_segment()
            st.rerun()

        else:
            st.error("Please fill in all the fields before starting the adventure.")
else:
    # Display the storyline so far
    for message in st.session_state.storyline:
        role = message['role']
        content = message['content']
        if role == 'assistant':
            st.markdown(f"**Narrator:** {content}")
        elif role == 'user':
            st.markdown(f"**You:** {content}")

    # Restart Adventure button
    if st.button("Restart Adventure", key='restart_button'):
        # Reset all relevant session state variables
        for key in ['storyline', 'interaction_count', 'chapter', 'game_stage', 'current_question', 'current_setting', 'current_choices', 'grade_level', 'selected_textbook', 'selected_novel']:
            st.session_state.pop(key, None)
        st.rerun()

    # Input box for user response
    user_input = st.text_input("Your response:", key='user_input')

    # Button to submit the response
    if st.button("Submit", key='submit_button') and user_input:
        # Append user's input to the storyline
        st.session_state.storyline.append({'role': 'user', 'content': user_input})
        # Generate the next part of the story
        generate_next_story_segment(user_input)
        # Clear the input box
        st.rerun()
