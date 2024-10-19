import os
import streamlit as st
import re
import time
import random
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
def query_pinecone(index, embedding, top_k=5, namespace=None):
    """Queries the Pinecone index for vectors similar to the provided query vector.

    Args:
        index (pinecone.Index): The Pinecone index to query.
        embedding (list): The query embedding vector.
        top_k (int): The number of top results to retrieve.
        namespace (str): The namespace to query within (if any).

    Returns:
        dict: The query results.
    """
    # Query the Pinecone index for similar vectors
    query_results = index.query(
        vector=embedding,  # The query vector as a list of floats
        top_k=top_k,
        include_values=False,     # Set to True if you need the vector values
        include_metadata=True,    # Include metadata in the response
        namespace=namespace       # Specify namespace if used
    )
    return query_results


def load_textbook_documents(textbook_name):
    directory_path = f"/home/polabs2/Code/RPG_teacher/data/textbooks/{textbook_name}"
    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return []
    try:
        documents = SimpleDirectoryReader(directory_path).load_data()
        if not documents:
            print(f"Warning: No documents found in '{directory_path}'.")
        return documents
    except Exception as e:
        print(f"Exception occurred while loading textbook documents: {e}")
        return []


def load_novel_documents(novel_name):
    directory_path = f"/home/polabs2/Code/RPG_teacher/data/novels/{novel_name}"
    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return []
    try:
        documents = SimpleDirectoryReader(directory_path).load_data()
        if not documents:
            print(f"Warning: No documents found in '{directory_path}'.")
        return documents
    except Exception as e:
        print(f"Exception occurred while loading novel documents: {e}")
        return []



def describe_adventure(textbook_name, textbook_chapter, novel_name):
    """
    Generates an adventure description combining themes from the novel with key topics from the textbook.
    """
    print("\n--- describe_adventure called ---")
    print(f"textbook_name: {textbook_name}")
    print(f"textbook_chapter: {textbook_chapter}")
    print(f"novel_name: {novel_name}")

    # Prepare the system prompt message
    # In describe_adventure
    system_prompt = (
        "You are an assistant that creates educational RPG adventures that combine fantasy novels with educational textbooks. "
        "Your task is to generate a few paragraphs describing the adventure setting and characters involved that combine themes from the novel with key topics from the textbook using the additional context provided below. "
        "Also, provide five places, events, or encounters that can be used to stage questions, listed separately under 'Places, Events, or Encounters'. "
        "Please format your response as:\n\n"
        "Adventure Description:\n[Your description here]\n\n"
        "Places, Events, or Encounters:\n1. [First place]\n2. [Second place]\n3. [Third place]\n4. [Fourth place]\n5. [Fifth place]"
    )

    # Mapping between textbook/novel names and index names
    index_name_mappings = {
        'Digital Marketing': 'digital-marketing-index',
        'The Hobbit': 'hobbit-index',
        # Add more mappings as needed
    }

    # Get the index names from the mappings
    index_name_textbook = index_name_mappings.get(textbook_name)
    index_name_novel = index_name_mappings.get(novel_name)

    # Check if index names are found
    if not index_name_textbook:
        print(f"Error: No index found for textbook '{textbook_name}'.")
        return None
    if not index_name_novel:
        print(f"Error: No index found for novel '{novel_name}'.")
        return None

    # Initialize the indexes
    textbook_index = pc.Index(name=index_name_textbook)
    novel_index = pc.Index(name=index_name_novel)

    top_k = 5  # You can adjust this value as needed

    # Retrieve content from the textbook related to the chapter
    start_time = time.time()
    textbook_query = f"{textbook_chapter}"
    # Get embedding for the textbook query
    textbook_embedding = get_embedding(textbook_query)
    textbook_response = query_pinecone(
        index=textbook_index,
        embedding=textbook_embedding,
        top_k=top_k
    )

    hobbit_chapter_summaries = {
        "Chapter 1: An Unexpected Party": "Bilbo Baggins, a hobbit, is visited by Gandalf, who invites him to join a group of dwarves on a journey to reclaim their homeland from the dragon Smaug. The dwarves, led by Thorin Oakenshield, arrive at Bilbo’s home and explain their quest to the Lonely Mountain. Bilbo is reluctantly recruited as their burglar.",
        "Chapter 2: Roast Mutton": "The company sets off and encounters trouble when they find three trolls. The trolls capture them, but Gandalf tricks the trolls into arguing until the sun rises, turning them to stone. The group finds treasure in the trolls’ lair.",
        "Chapter 3: A Short Rest": "The group travels to Rivendell, the home of the Elves, where they receive guidance and rest. Elrond, the elf lord, helps them decipher the map and runes, revealing the secret entrance to the Lonely Mountain.",
        "Chapter 4: Over Hill and Under Hill": "The group crosses the Misty Mountains but is captured by goblins in the caves. Gandalf helps them escape, but during the escape, Bilbo gets separated from the others.",
        "Chapter 5: Riddles in the Dark": "Bilbo encounters Gollum, a strange creature living in the goblin caves. They engage in a riddle contest, and Bilbo wins. He discovers a magical ring that makes him invisible and uses it to escape Gollum and rejoin the dwarves.",
        "Chapter 6: Out of the Frying-Pan into the Fire": "Bilbo and the dwarves escape the goblins, but they are soon pursued by Wargs (evil wolves). They are rescued by eagles who carry them to safety.",
        "Chapter 7: Queer Lodgings": "The group seeks shelter with Beorn, a shape-shifter who can turn into a bear. He provides them with provisions and advice for their journey through Mirkwood.",
        "Chapter 8: Flies and Spiders": "In Mirkwood, the group faces dangers including giant spiders, which capture them. Bilbo uses the ring and his newfound courage to rescue the dwarves. They are later captured by Wood Elves and imprisoned.",
        "Chapter 9: Barrels Out of Bond": "Bilbo helps the dwarves escape the Wood Elves by hiding them in barrels that are floated down the river to Lake-town (Esgaroth), a human settlement near the Lonely Mountain.",
        "Chapter 10: A Warm Welcome": "The group arrives in Lake-town, where they are warmly received by the people. The townspeople hope that Thorin will fulfill the prophecy of defeating Smaug and restoring prosperity.",
        "Chapter 11: On the Doorstep": "The group reaches the Lonely Mountain and locates the secret entrance. They cannot open the door at first but finally figure out how to use the map’s clues to unlock it.",
        "Chapter 12: Inside Information": "Bilbo enters the dragon’s lair and steals a cup. He later converses with Smaug, learning about the dragon’s vulnerabilities. Smaug becomes enraged and flies off to attack Lake-town.",
        "Chapter 13: Not at Home": "With Smaug gone, the dwarves explore the treasure hoard in the Lonely Mountain. Bilbo discovers the Arkenstone, a precious gem, but keeps it hidden from Thorin.",
        "Chapter 14: Fire and Water": "Smaug attacks Lake-town, but Bard, a human archer, kills him with an arrow aimed at a weak spot. The people of Lake-town begin to rebuild, and Bard claims his rightful share of the treasure.",
        "Chapter 15: The Gathering of the Clouds": "Thorin hears of Smaug’s death and becomes obsessed with defending the treasure. Elves and men approach the mountain, seeking a share of the hoard. Thorin refuses to negotiate, leading to a standoff.",
        "Chapter 16: A Thief in the Night": "Bilbo, hoping to prevent conflict, secretly gives the Arkenstone to Bard and the Elvenking, hoping they can use it as leverage in negotiations with Thorin.",
        "Chapter 17: The Clouds Burst": "Thorin refuses to negotiate even after seeing the Arkenstone. A battle seems inevitable, but just then, goblins and Wargs arrive, forcing the men, elves, and dwarves to unite against a common enemy in the Battle of Five Armies.",
        "Chapter 18: The Return Journey": "Thorin is fatally wounded in battle but reconciles with Bilbo before he dies. After the battle, Bilbo declines a large share of the treasure and returns home with only a small portion.",
        "Chapter 19: The Last Stage": "Bilbo returns to his hobbit-hole in the Shire to find that he has been presumed dead. He settles back into a quiet life, forever changed by his adventures."
    }

    # Use the existing hobbit_chapter_summaries dictionary
    chapter_keys = list(hobbit_chapter_summaries.keys())
    # Select two random consecutive chapters
    start_idx = random.randint(0, len(chapter_keys) - 2)  # Ensure there's a next chapter
    consecutive_chapters = chapter_keys[start_idx:start_idx + 2]
    # Combine the summaries for the query
    combined_summary = (
            hobbit_chapter_summaries[consecutive_chapters[0]] + " " + hobbit_chapter_summaries[consecutive_chapters[1]]
    )

    # Get embedding for the novel query
    # Query the novel index
    novel_embedding = get_embedding(combined_summary)
    novel_response = query_pinecone(
        index=novel_index,
        embedding=novel_embedding,
        top_k=top_k
    )
    #print(f'length novel response: {len(novel_response)}')

    # Check if matches are found
    if not textbook_response.matches:
        print("No matches found in textbook_response.")
    else:
        print(f"Found {len(textbook_response.matches)} matches in textbook_response.")

    if not novel_response.matches:
        print("No matches found in novel_response.")
    else:
        print(f"Found {len(novel_response.matches)} matches in novel_response.")

    # Extract content from the retrieved documents
    textbook_context_docs = [
        match.metadata['text'] for match in textbook_response.matches
        if match.metadata and 'text' in match.metadata
    ]
    novel_context_docs = [
        match.metadata['text'] for match in novel_response.matches
        if match.metadata and 'text' in match.metadata
    ]

    # Combine the retrieved content
    retrieved_content = "\n".join(textbook_context_docs + novel_context_docs)
    print(f'combined content length: {len(retrieved_content)}')

    # Create the user message
    user_message = (
        f"Textbook: {textbook_name}\n"
        f"Chapter: {textbook_chapter}\n"
        f"Novel: {novel_name}\n\n"
        f"Retrieved Content:\n{retrieved_content}"
    )

    # Prepare messages for the LLM
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=MessageRole.USER, content=user_message)
    ]

    # Call the LLM to generate the adventure description
    response = llm_predict_with_retry(messages)

    response_content = response.message.content.strip()
    adventure_match = re.search(r"Adventure Description:\s*(.*?)\s*Places, Events, or Encounters:", response_content,re.DOTALL)
    places_match = re.search(r"Places, Events, or Encounters:\s*(.*)", response_content, re.DOTALL)

    if adventure_match and places_match:
        adventure_description = adventure_match.group(1).strip()
        places_text = places_match.group(1).strip()
        places_list = re.findall(r"\d+\.\s*(.*)", places_text)# Extract the list of places
    else:
        adventure_description = response_content        # Handle parsing error
        places_list = []

    # Store the places_list in session state for later use
    st.session_state['places_events_encounters'] = places_list
    st.session_state['places_events_encounters'] = [
        limit_to_two_sentences(desc) for desc in st.session_state['places_events_encounters']
    ]

    print(f"Generated adventure description: {adventure_description}")
    print("--- describe_adventure ended ---\n")
    return adventure_description


def describe_setting(text, novel, adventure_description, place_event_encounter):
    """
    Generates a vivid setting description that combines textbook content and novel elements.
    Uses top_k relevant documents from both indexes as context.
    Limits the description to exactly 10 sentences, split into two paragraphs.
    """
    print("\n--- describe_setting called ---")
    print(f"text: {text[:100]}...")
    print(f"novel: {novel[:100]}...")

    # Mapping between textbook/novel names and index names
    index_name_mappings = {
        'Digital Marketing': 'digital-marketing-index',
        'The Hobbit': 'hobbit-index',
    }

    # Get the textbook and novel names from session state
    textbook_name = st.session_state.selected_textbook
    novel_name = st.session_state.selected_novel

    # Get the index names from the mappings
    index_name_textbook = index_name_mappings.get(textbook_name)
    index_name_novel = index_name_mappings.get(novel_name)

    # Check if index names are found
    if not index_name_textbook:
        print(f"Error: No index found for textbook '{textbook_name}'.")
        return None
    if not index_name_novel:
        print(f"Error: No index found for novel '{novel_name}'.")
        return None

    # Initialize the indexes
    textbook_index = pc.Index(name=index_name_textbook)
    novel_index = pc.Index(name=index_name_novel)

    top_k = 5  # Default value

    # Generate embeddings for the queries
    textbook_embedding = get_embedding(text)
    novel_embedding = get_embedding(novel)

    # Query both indexes
    textbook_response = query_pinecone(textbook_index, textbook_embedding, top_k=top_k)
    novel_response = query_pinecone(novel_index, novel_embedding, top_k=top_k)

    # Extract content from the retrieved documents
    textbook_context_docs = [match.metadata['text'] for match in textbook_response.matches if match.metadata and 'text' in match.metadata]
    novel_context_docs = [match.metadata['text'] for match in novel_response.matches if match.metadata and 'text' in match.metadata]

    # Combine the context documents
    context = "\n".join(textbook_context_docs + novel_context_docs)

    # Use the specified system prompt
    system_prompt = (
        "You are an assistant that generates vivid and engaging setting descriptions that combine elements from fantasy novels with educational textbooks. "
        f"Your task is to seamlessly continue the story from the adventure's current point, describing the following place, event, or encounter: '{place_event_encounter}'. "
        "Ensure that the narrative flows naturally from the end of the provided adventure description. Do not repeat any content from the adventure description, but instead pick up from where it left off. "
        "Provide a vivid description of the setting, characters, and action in exactly 5 sentences. Integrate key concepts from the textbook into the scene without explicitly stating the number of sentences."
        " Avoid phrases such as 'Here are exactly 5 sentences' or 'Narrator says...'. Simply provide the descriptive text."
    )

    # Construct the prompt
    prompt = f"""
    Continue the adventure from where the following description leaves off. Ensure the setting description smoothly picks up the narrative without repeating any details. 
    Integrate key textbook concepts into the scene naturally. Generate exactly 5 sentences to describe the current encounter or place.

    Adventure Description:
    {adventure_description}

    Context:
    {context}
    """

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=MessageRole.USER, content=prompt)
    ]

    # Call the LLM and set the max_tokens limit
    response = llm_predict_with_retry(messages)#, max_tokens=max_tokens)
    setting_description = response.message.content.strip()

    sentences = re.split(r'(?<=[.!?]) +', setting_description)  # Split by sentence boundaries
    # Split into two paragraphs (5 sentences per paragraph)
    paragraph_1 = ' '.join(sentences[:5])
    paragraph_2 = ' '.join(sentences[5:10])

    final_description = f"{paragraph_1}\n\n{paragraph_2}"  # Two paragraphs with a line break

    print(f"Final setting description (10 sentences in two paragraphs): {final_description}")
    print("--- describe_setting ended ---\n")

    return final_description


def describe_question(setting_description, text, novel):
    """
    Generates a challenging question based on the setting description, text, and novel content.
    Includes relevant questions from chapter_summary_notes.csv, paired by the 'key' column.
    """
    print("\n--- describe_question called ---")
    print(f"Setting description: {setting_description[:100]}...")

    # Mapping between textbook/novel names and index names
    index_name_mappings = {
        'Digital Marketing': 'digital-marketing-index',
        'The Hobbit': 'hobbit-index',
    }

    # Get the textbook and novel names from session state
    textbook_name = st.session_state.selected_textbook
    novel_name = st.session_state.selected_novel

    # Get the index names from the mappings
    index_name_textbook = index_name_mappings.get(textbook_name)
    index_name_novel = index_name_mappings.get(novel_name)

    # Check if index names are found
    if not index_name_textbook:
        print(f"Error: No index found for textbook '{textbook_name}'.")
        return None, None
    if not index_name_novel:
        print(f"Error: No index found for novel '{novel_name}'.")
        return None, None

    # Initialize the indexes
    textbook_index = pc.Index(name=index_name_textbook)
    novel_index = pc.Index(name=index_name_novel)

    top_k = 5  # Default value

    # Generate embeddings for the queries
    textbook_embedding = get_embedding(text)
    novel_embedding = get_embedding(novel)

    # Query both indexes
    textbook_response = query_pinecone(textbook_index, textbook_embedding, top_k=top_k)
    novel_response = query_pinecone(novel_index, novel_embedding, top_k=top_k)

    # Extract content from the retrieved documents
    # Use dot notation to access attributes
    textbook_context_docs = [
        match.metadata['text'] for match in textbook_response.matches
        if match.metadata and 'text' in match.metadata
    ]
    novel_context_docs = [
        match.metadata['text'] for match in novel_response.matches
        if match.metadata and 'text' in match.metadata
    ]

    # Combine the context documents
    context = "\n".join(textbook_context_docs + novel_context_docs)

    # Retrieve relevant rows for the current chapter from chapter_summary_notes.csv
    df = st.session_state.chapter_summary_notes
    chapter = st.session_state.chapter
    textbook_name_lower = textbook_name.lower()
    novel_name_lower = novel_name.lower()

    # Filter for questions related to the textbook and the novel for the current chapter
    questions_df = df[
        (df['chapter'] == chapter) &
        (df['data_type'] == 'question') &
        ((df['document'] == textbook_name_lower) | (df['document'] == novel_name_lower))
    ]

    # Pair questions and answers using the 'key' column
    questions_dict = {}
    for key in questions_df['key'].unique():
        question_data = questions_df[questions_df['key'] == key]
        question_row = question_data[question_data['document_type'] == 'text_book']
        answer_row = question_data[question_data['document_type'] == 'fantasy_novel']
        if not question_row.empty and not answer_row.empty:
            question = question_row['text'].values[0]
            answer = answer_row['text'].values[0]
            questions_dict[question] = answer

    # Randomly select a question-answer pair (or you can cycle through them in sequence)
    import random
    if questions_dict:
        sample_question, sample_answer = random.choice(list(questions_dict.items()))
    else:
        sample_question = "No questions available for this chapter."
        sample_answer = "No answer available."

    print(f'sample q: {sample_question}')

    # Combine the context from both indexes and integrate the setting description
    prompt = f"""
    Using the following setting description, context, and sample question, create an engaging question for the player that ties into the RPG adventure. 
    The question should integrate educational content from the textbook. Please provide the question and its answer.

    Setting Description:
    {setting_description}

    Context:
    {context}

    Sample Question:
    {sample_question}

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


def give_hint(question, answer):
    """
    Provides a hint to the user using RAG results from the textbook index.
    Uses the answer to point the user in the correct direction without directly giving them the answer.
    """
    print("\n--- give_hint called ---")
    print(f"Question: {question}")

    # Mapping between textbook names and index names
    index_name_mappings = {
        'Digital Marketing': 'digital-marketing-index',
        # Add more mappings as needed
    }

    # Get the textbook name from session state
    textbook_name = st.session_state.selected_textbook

    # Get the index name from the mappings
    index_name_textbook = index_name_mappings.get(textbook_name)

    # Check if index name is found
    if not index_name_textbook:
        print(f"Error: No index found for textbook '{textbook_name}'.")
        return None

    # Initialize the index
    textbook_index = pc.Index(name=index_name_textbook)

    top_k = 5  # Default value

    # Generate embedding for the question
    question_embedding = get_embedding(question)

    # Query the textbook index
    textbook_response = query_pinecone(
        index=textbook_index,
        embedding=question_embedding,
        top_k=top_k
    )

    # Extract matched IDs
    match_ids = [match.id for match in textbook_response.matches]

    # Fetch metadata for matched IDs
    fetch_response = textbook_index.fetch(ids=match_ids)

    # Extract text from metadata
    context_docs = [
        vector.metadata['text'] for vector in fetch_response.vectors.values()
        if vector.metadata and 'text' in vector.metadata
    ]

    # Combine the context documents
    context = "\n".join(context_docs)

    # Construct the prompt
    prompt = f"""
    Using the following context and the provided answer, provide a hint to help the student answer the question. Do not provide the answer directly, but guide them towards the correct answer.

    Question:
    {question}

    Context:
    {context}

    Answer:
    {answer}

    Please provide a helpful hint without giving away the answer.
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
    Assesses the user's answer based on the provided rubric and provides feedback.
    """
    print("\n--- grade_answer called ---")
    print(f"user_answer: {user_answer}")
    print(f"question: {question}")
    print(f"correct_answer: {correct_answer}")

    # Define the rubric
    rubric = """
    4: Clearly demonstrates an understanding of the task, completes all requirements, and provides an insightful explanation or opinion of the text, or extends aspects of the text.
    3: Demonstrates an understanding of the task, completes all requirements, and provides some explanation or opinion using situations or ideas from the text as support.
    2: May address all of the requirements, but demonstrates only a partial understanding of the task and uses text incorrectly or with limited success resulting in an inconsistent or flawed explanation.
    1: Demonstrates minimal understanding of the task, does not complete all requirements, and provides only a vague reference to, or no use of, the text.
    0: Is completely irrelevant or off-topic.
    """

    prompt = f"""
    As an educator, assess the following student's answer to the question based on the provided rubric. Use the rubric to determine the level (0-4) that best describes the student's answer. Provide the level and an explanation, referencing the student's answer and the rubric.

    Rubric:
    {rubric}

    Question:
    {question}

    Correct Answer:
    {correct_answer}

    Student's Answer:
    {user_answer}

     Please respond with:

    Grade: [Your grade here]
    Feedback: [Your feedback here]
    """

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are an expert educator assessing a student's answer."),
        ChatMessage(role=MessageRole.USER, content=prompt)
    ]
    response = llm_predict_with_retry(messages)
    feedback = response.message.content.strip()

    grade_match = re.search(r"Grade:\s*(\d+)", feedback)
    feedback_match = re.search(r"Feedback:\s*(.*)", feedback, re.DOTALL)

    if grade_match and feedback_match:
        grade = int(grade_match.group(1))
        feedback_text = feedback_match.group(1).strip()
    else:
        grade = None
        feedback_text = "Unable to parse feedback."
    print({
        "grade": grade,
        "feedback": feedback_text
    })
    # Return structured result
    return {
        "grade": grade,
        "feedback": feedback_text
    }




def player_choice(choices):
    """
    Presents remaining choices to the player and returns a formatted string for display.
    """
    choices_prompt = "Where would you like to go next?\n"
    for idx, choice in enumerate(choices, start=1):
        choices_prompt += f"{idx}. {choice}\n"
    return choices_prompt


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

        response = Settings.llm.chat(messages=messages)#, max_tokens=300)
        return response
    except Exception as e:
        print(f"Exception: {e}")
        raise e


def limit_to_two_sentences(text):
    """
    Limit the provided text to two sentences.
    """
    # Split the text into sentences using regular expressions
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Return the first two sentences joined back together
    #return ' '.join(sentences[:2])
    return sentences[0]


def generate_next_story_segment(user_input=None):
    """
    Main function to control the flow of the RPG adventure.
    It starts the adventure, handles answers, presents choices, and continues the loop.
    """
    print("\n--- generate_next_story_segment called ---")
    print(f"user_input: {user_input}")

    # Debugging session state
    print(f"Current game stage: {st.session_state.get('game_stage', 'Not set')}")
    print(f"Places, Events, Encounters: {st.session_state.get('places_events_encounters', [])}")

    # Initialize game state if not already done
    if 'game_stage' not in st.session_state:
        st.session_state.game_stage = 'start'
        st.session_state.current_question = ''
        st.session_state.current_question_answer = ''
        st.session_state.current_setting = ''
        st.session_state.current_choices = []
        st.session_state.chapter = 1
        # Initialize the places, events, encounters
        st.session_state['places_events_encounters'] = ['Forest', 'Ruins', 'Mountain Pass', 'Old Bridge', 'Tavern']

    if 'storyline' not in st.session_state:
        st.session_state.storyline = []

    # Placeholder for the storyline
    story_placeholder = st.empty()

    # Only update the storyline incrementally
    if st.session_state.game_stage == 'start':
        print("Game stage: start")

        # Load the selected textbook and novel content
        text = st.session_state.selected_textbook
        novel = st.session_state.selected_novel
        chapter = st.session_state.chapter

        # Generate the adventure description and store it
        adventure_description = describe_adventure(textbook_name=text, textbook_chapter=chapter, novel_name=novel)
        st.session_state.current_setting = adventure_description
        st.session_state.storyline.append({'role': 'assistant', 'content': adventure_description})

        # Display the adventure description immediately
        with story_placeholder.container():
            for message in st.session_state.storyline:
                role = message['role']
                content = message['content']
                if role == 'assistant':
                    st.markdown(f"**Narrator:** {content}")
                elif role == 'user':
                    st.markdown(f"**You:** {content}")

        # Proceed with the next step after showing the adventure description
        # Extract the first 'place, event, or encounter' and continue
        if 'places_events_encounters' in st.session_state and st.session_state['places_events_encounters']:
            place_event_encounter = st.session_state['places_events_encounters'][0]  # Do not pop yet
        else:
            place_event_encounter = "A generic setting"

        # Call describe_setting to describe the first setting
        setting_description = describe_setting(text, novel, adventure_description, place_event_encounter)
        st.session_state.current_setting = st.session_state.current_setting + ' ' + setting_description
        st.session_state.storyline.append({'role': 'assistant', 'content': setting_description})

        # Generate the first question
        question, question_answer = describe_question(setting_description, text, novel)
        st.session_state.current_question = question
        st.session_state.current_question_answer = question_answer
        st.session_state.storyline.append({'role': 'assistant', 'content': question})

        # Update game stage to awaiting_answer
        st.session_state.game_stage = 'awaiting_answer'

    elif st.session_state.game_stage == 'awaiting_answer':
        print("Game stage: awaiting_answer")

        if user_input is None:
            print("No user input received.")
            return

        # Handle user's answer
        user_answer = user_input
        st.session_state.storyline.append({'role': 'user', 'content': user_answer})

        # Grade the answer and get structured result
        grading_result = grade_answer(user_answer, st.session_state.current_question,
                                      st.session_state.current_question_answer)

        # Add feedback to the storyline
        st.session_state.storyline.append({'role': 'assistant', 'content': grading_result['feedback']})

        # If a valid grade is returned, present the remaining choices
        if grading_result['grade'] is not None:
            # Check if there are any remaining places, events, or encounters
            if st.session_state['places_events_encounters']:
                # Present remaining choices to the player
                st.session_state.game_stage = 'awaiting_choice'  # Set the state to awaiting_choice
            else:
                # If no more places left, end the game
                st.session_state.storyline.append(
                    {'role': 'assistant', 'content': "You have completed all encounters!"})
                st.session_state.game_stage = 'end'

    elif st.session_state.game_stage == 'awaiting_choice':
        print("Game stage: awaiting_choice")

        # Display buttons for the available choices
        if st.session_state['places_events_encounters']:
            st.markdown("### Choose where to go next:")

            # Temporary variable to track selected choice
            selected_choice = None

            for idx, choice in enumerate(st.session_state['places_events_encounters']):
                if st.button(f"{choice}", key=f"choice_{idx}"):
                    selected_choice = idx
                    break  # Exit loop after selection

            # Handle the selected choice after the loop
            if selected_choice is not None:
                # Remove the selected place from the list and process it
                chosen_place = st.session_state['places_events_encounters'].pop(selected_choice)

                # Use the chosen place in the next setting description
                text = st.session_state.selected_textbook
                novel = st.session_state.selected_novel
                adventure_description = st.session_state.current_setting
                setting_description = describe_setting(text, novel, adventure_description, chosen_place)
                st.session_state.current_setting = setting_description
                st.session_state.storyline.append({'role': 'assistant', 'content': setting_description})

                # Generate the next question for this place
                question, question_answer = describe_question(setting_description, text, novel)
                st.session_state.current_question = question
                st.session_state.current_question_answer = question_answer
                st.session_state.storyline.append({'role': 'assistant', 'content': question})

                # Update game stage to awaiting_answer
                st.session_state.game_stage = 'awaiting_answer'
                st.rerun()  # Rerun to update the UI after choice

        # Keep track of the number of messages already displayed
    if 'last_displayed_message_index' not in st.session_state:
        st.session_state.last_displayed_message_index = 0

        # Display only new messages
    new_messages = st.session_state.storyline[st.session_state.last_displayed_message_index:]
    with st.container():
        for message in new_messages:
            role = message['role']
            content = message['content']
            if role == 'assistant':
                st.markdown(f"**Narrator:** {content}")
            elif role == 'user':
                st.markdown(f"**You:** {content}")

    # Update the index
    st.session_state.last_displayed_message_index = len(st.session_state.storyline)


# Initialize session state variables
if 'storyline' not in st.session_state:
    st.session_state.storyline = []
if 'game_stage' not in st.session_state:
    st.session_state.game_stage = 'start'
if 'chapter' not in st.session_state:
    st.session_state.chapter = 'Chapter 7: Web Development and Design'
if 'interaction_count' not in st.session_state:
    st.session_state.interaction_count = 0
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


# Textbook and novel options
textbook_options = ['Digital Marketing']  # Add more textbooks as needed
novel_options = ['The Hobbit']  # Add more novels as needed

# Chapter options
chapter_options = [
    'Chapter 7: Web Development and Design',
    'Chapter 12: Video Content Creation',
    'Chapter 13: Social Media'
]

# Ensure session state values are valid
if 'selected_textbook' not in st.session_state or st.session_state.selected_textbook not in textbook_options:
    st.session_state.selected_textbook = textbook_options[0]

if 'selected_novel' not in st.session_state or st.session_state.selected_novel not in novel_options:
    st.session_state.selected_novel = novel_options[0]

if 'chapter' not in st.session_state or st.session_state.chapter not in chapter_options:
    st.session_state.chapter = chapter_options[0]

st.title("Interactive RPG Adventure")
st.write("Embark on a journey that combines fantasy storytelling with educational challenges!")

# Input for grade level
if st.session_state.game_stage == 'start' and not st.session_state.storyline:
    st.selectbox("Select the textbook:", textbook_options, key='selected_textbook')
    st.selectbox("Select the chapter:", chapter_options, key='chapter')
    st.selectbox("Select the novel:", novel_options, key='selected_novel')

    # Button to start the adventure
    if st.button("Start Adventure", key='start_button'):
        if st.session_state.selected_textbook and st.session_state.selected_novel and st.session_state.chapter:
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

    # Input box for user response if the game is awaiting an answer
    if st.session_state.game_stage == 'awaiting_answer':
        user_input = st.text_input("Your response:", key='user_input')

        if st.button("Submit"):
            if user_input:
                # Process the user's answer
                generate_next_story_segment(user_input=user_input)
                st.rerun()
            else:
                st.error("Please enter a response.")
    elif st.session_state.game_stage == 'awaiting_choice':
        # Call generate_next_story_segment without user input to display choices
        generate_next_story_segment()
    elif st.session_state.game_stage == 'end':
        st.markdown("**Game Over:** You have completed all encounters!")
        if st.button("Restart Adventure", key='restart_button'):
            # Reset all relevant session state variables
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()