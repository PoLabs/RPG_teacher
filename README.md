# RPG-Based Educational App

This project is an RPG-style educational app that combines textbook content with fantasy adventure themes, creating an immersive, interactive learning experience. Users can explore fantasy settings filled with educational challenges and trivia based on chosen textbook material.

---

## Demo Links
- **[RPG Teacher](https://rpg-teacher.streamlit.app/)**: Live app hosted on streamlit.
- **[Google Drive - Download Repository](https://drive.google.com/drive/folders/1t2hQd7eYWozDv-BpTJ-pVMc8D4lbyq-m?usp=sharing)**: Download the project repository.
- **[LinkedIn Post](https://www.linkedin.com/posts/vincent-pisano-78674634_not-found-activity-7260403151684407296-Toka?utm_source=share&utm_medium=member_desktop)**: Learn more about the project and community feedback.
- **[Video Tutorial](https://www.veed.io/view/eafe6942-63e6-4fdd-bd00-d70e30e27e64?panel=share)**: Watch a full walkthrough of the app and its features.

---

## Installation Instructions

To set up the app, follow these steps:

### Step 1: Download and Unpack
1. Download the zip file from the Google Drive link provided.
2. Unpack the zip file:
```bash
unzip RPG_Teacher.zip -d RPG_Teacher
```
3. Navigate to the project directory:
```bash
cd RPG_Teacher
```
### Step 2:  Set Up a Virtual Environment
1. Create and activate a virtual environment:
```bash
python3 -m venv venv_app
source venv_app/bin/activate  # On Windows, use venv_app\Scripts\activate
pip install -r requirements.txt
```
### Step 3: Configure API Keys
1. Edit the API keys in .streamlit/secrets.toml if needed:
```bash
nano .streamlit/secrets.toml
```
### Step 4: Run the App
1. Start the app using Streamlit:
```bash
streamlit run rpg_streamlit.py
```



## Adding New Documents

To add a new textbook or novel, follow these steps:

### Step 1: Set Up Curator Environment
Create and activate a new virtual environment for the curator:
```bash
python3 -m venv venv_curator
source venv_curator/bin/activate
```
Install the dependencies from requirements_curator.txt:
```bash
pip install -r requirements_curator.txt
```
### Step 2: Convert Raw PDF Document
Place your raw PDF document in data/raw.
Run the conversion script:
```bash
python nemo_curator_tutorial.py --input-data-dir /data/raw --output-clean-dir /data/out --input-file-type pdf --output-file-type jsonl --batch-size 64
```
### Step 3: Organize Processed Files
Create new folders in data/novels or data/textbooks for each new document.
Manually copy the JSONL files from data/out into the newly created folder. Note that data/out will contain JSONL files from all processed documents.
### Step 4: Upload Data to Pinecone
Open and run the Jupyter notebook load_pinecone.ipynb.
Modify lines 14-15 and lines 70/72 to specify the names of your new document indexes.
If you’re only uploading either a textbook or a novel, comment out the relevant lines (70 or 72) in the notebook.
Note: For large documents, this upload process may take 15-20 minutes.

### Step 5: Update the Application Code
Open rpg_streamlit.py.
Update the index_name_mappings dictionary (line 42) with the new document index names.
Add the new documents to the Streamlit UI section (lines 1093-1117).

### Step 6: Switch back to venv_app and run
```bash
source venv_curator/bin/activate
streamlit run rpg_streamlit.py
```


