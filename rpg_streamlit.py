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
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


# set up model, index, embedding, nvidia API -------------------------------------------------------------------------
TOP_K = 5  # Global constant

# Set NVIDIA API Key (Hard-coded)
nvidia_api_key = "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"  # Replace with your actual NVIDIA API key
os.environ["NVIDIA_API_KEY"] = nvidia_api_key
assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
print(f"NVIDIA API key is set: {nvidia_api_key[:5]}...")  # Masking for display

# Initialize the NVIDIA LLM using Settings
Settings.llm = NVIDIA(model="meta/llama-3.1-70b-instruct")#llama-3.1-405b-instruct")
# Initialize the NVIDIA embedding model
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
# Set the text splitter
Settings.text_splitter = SentenceSplitter(chunk_size=400)


api_key = "d82b0e3a-acd5-4197-a10c-84245c2f9331"  # Replace with your actual Pinecone API key
pc = Pinecone(api_key=api_key)

openai_api_key = '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'
client = OpenAI(api_key=openai_api_key)

index_name_textbook = 'digital-marketing-index'
index_name_novel = 'hobbit-index'

textbook_index = pc.Index(name=index_name_textbook)
novel_index = pc.Index(name=index_name_novel)


def get_embedding(text):
    """Generate embeddings using OpenAI API."""
    response = client.embeddings.create(model="text-embedding-ada-002", input=text)
    # Access the first embedding from the response using dot notation
    embedding = response.data[0].embedding
    return embedding

# Function to query Pinecone index
def query_pinecone(index, embedding, top_k=5):
    query_results = index.query(vector=embedding, top_k=top_k)
    return query_results

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
    First paragraph should be about the setting. Second and third paragraphs should integrate textbook concepts into the setting.
    Please provide a detailed setting description that blends these elements seamlessly using the context provided below.

    Context:
    {context}
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
    Includes relevant questions from chapter_summary_notes.csv, paired by the 'key' column.
    """
    print("\n--- describe_question called ---")
    print(f"Setting description: {setting_description[:100]}...")

    # Access indexes from st.session_state
    textbook_index = st.session_state.textbook_index
    novel_index = st.session_state.novel_index
    top_k = 5  # Default value

    # Retrieve relevant rows for the current chapter from chapter_summary_notes.csv
    df = st.session_state.chapter_summary_notes
    chapter = st.session_state.chapter
    textbook_name = st.session_state.selected_textbook.lower()
    novel_name = st.session_state.selected_novel.lower()

    # Filter for questions related to the textbook and the novel for the current chapter
    questions_df = df[
        (df['chapter'] == chapter) &
        (df['data_type'] == 'question') &
        ((df['document'] == textbook_name) | (df['document'] == novel_name))
    ]

    # Pair questions and answers using the 'key' column
    questions_dict = {}
    for key in questions_df['key'].unique():
        question_data = questions_df[questions_df['key'] == key]
        question = question_data[question_data['document_type'] == 'text_book']['text'].values[0]
        answer = question_data[question_data['document_type'] == 'fantasy_novel']['text'].values[0]
        questions_dict[question] = answer

    # Randomly select a question-answer pair (or you can cycle through them in sequence)
    import random
    if questions_dict:
        question, answer = random.choice(list(questions_dict.items()))
    else:
        question = "No questions available for this chapter."
        answer = "No answer available."

    # Combine the context from both indexes (textbook and novel) and integrate the setting description
    prompt = f"""
    Using the following setting description, context, and sample question, create an engaging question for the player that ties into the RPG adventure. The question should be appropriate for grade level {st.session_state.grade_level} and integrate educational content from the textbook. Please provide the question and its answer.

    Setting Description:
    {setting_description}

    Context:
    {text}

    Question:
    {question}

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


def get_textbook_content(textbook_name, chapter_number):
    documents = load_textbook_documents(textbook_name)
    for doc in documents:
        content = doc.get_content()
        if f"Chapter {chapter_number}" in content:
            return content
    return "Textbook content not found."

def get_novel_content(novel_name, chapter_number):
    documents = load_novel_documents(novel_name)
    for doc in documents:
        content = doc.get_content()
        if f"Chapter {chapter_number}" in content:
            return content
    return "Novel content not found."



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
            # Continue the journey: Prompt for the next chapter
            st.text("You've completed this chapter. Please select the next chapter to continue.")
            #st.session_state.chapter = st.selectbox("Select the next chapter:", chapter_options, key='next_chapter')

            # Load the selected textbook and novel content for the new chapter
            st.session_state.selected_textbook_content = get_textbook_content(
                st.session_state.selected_textbook, st.session_state.chapter)
            st.session_state.selected_novel_content = get_novel_content(
                st.session_state.selected_novel, st.session_state.chapter)

            # Generate the new setting description
            text = st.session_state.selected_textbook_content
            novel = st.session_state.selected_novel_content
            setting_description = describe_setting(text, novel)
            st.session_state.current_setting = setting_description
            st.session_state.storyline.append({'role': 'assistant', 'content': setting_description})

            # Generate the first question for the new chapter
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
    st.session_state.chapter = 7
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
    # Number of chapters (adjust this to the actual number of chapters in your content)
    chapter_options = [7,12,13]#list(range(1, 11))  # Chapters 1 to 10
    # Dropdown for chapter selection
    st.selectbox("Select the chapter:", chapter_options, key='chapter')

    # Dropdown for novel selection
    st.selectbox("Select the novel:", novel_options, key='selected_novel')

    # Button to start the adventure
    # After the user selects the textbook and novel and clicks "Start Adventure"
    if st.button("Start Adventure", key='start_button'):
        if st.session_state.grade_level and st.session_state.selected_textbook and st.session_state.selected_novel and st.session_state.chapter:
            # Load the selected textbook and novel content for the selected chapter
            st.session_state.selected_textbook_content = get_textbook_content(
                st.session_state.selected_textbook, st.session_state.chapter)
            st.session_state.selected_novel_content = get_novel_content(
                st.session_state.selected_novel, st.session_state.chapter)

            # Create indexes for the selected textbook and novel
            st.session_state.textbook_index = create_textbook_index(st.session_state.selected_textbook)
            st.session_state.novel_index = create_novel_index(st.session_state.selected_novel)

            # Start the game
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

    if st.button("Submit"):
        if user_input:
            # Generate embedding for query
            query_embedding = get_embedding(user_input)

            # Query the textbook and novel indexes
            textbook_results = query_pinecone(textbook_index, query_embedding)
            novel_results = query_pinecone(novel_index, query_embedding)

            # Display results
            st.write("Textbook Results:")
            st.json(textbook_results)

            st.write("Novel Results:")
            st.json(novel_results)
        else:
            st.error("Please enter a query.")
