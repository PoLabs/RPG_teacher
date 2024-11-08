import argparse
import os
import re
import fitz  # PyMuPDF for reading PDFs
import pandas as pd
import dask.dataframe as dd  # Import Dask DataFrame
from hashlib import md5  # For creating unique filenames

from nemo_curator.modifiers import DocumentModifier
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.unicode_reformatter import UnicodeReformatter
from nemo_curator.utils.distributed_utils import get_client, write_to_disk
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir, get_batched_files
from nemo_curator.utils.script_utils import add_distributed_args



# python nemo_curator_tutorial.py --input-data-dir /home/polabs2/Code/RPG_teacher/data/raw --output-clean-dir /home/polabs2/Code/RPG_teacher/data/out --input-file-type pdf --output-file-type jsonl --batch-size 64

# Custom Cleaner: Removes Unicode and special characters
class CustomTextCleaner(DocumentModifier):
    """Custom cleaner to remove Unicode and special characters."""

    def modify_document(self, text):
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
        cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)  # Remove special characters
        return cleaned_text


# PDF Document Iterator for reading PDFs and extracting text
class PDFDocumentIterator:
    """Iterator class for reading PDF files and extracting text."""

    def iterate(self, file_path):
        """Yields extracted text from each page of the PDF."""
        with fitz.open(file_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf.load_page(page_num)
                text = page.get_text()
                yield {"page_number": page_num}, text

# Generate a unique filename using MD5 hash of content + row index
def generate_unique_filename(filename, page_number, content, index):
    unique_hash = md5(f"{filename}_{page_number}_{content}_{index}".encode('utf-8')).hexdigest()
    return f"{filename}_page_{page_number}_{unique_hash}.jsonl"


# Main function: reads data from disk, processes, and writes back in NeMo Curator format
def main(args):
    # Make the output directories
    output_clean_dir = expand_outdir_and_mkdir(args.output_clean_dir)

    # Initialize the custom text cleaner and Unicode reformatter
    custom_cleaner = CustomTextCleaner()
    unicode_cleaner = UnicodeReformatter()

    for files in get_batched_files(
            args.input_data_dir,
            output_clean_dir,
            args.input_file_type,
            batch_size=args.batch_size,
    ):
        # Read and process PDFs
        dataset = []
        iterator = PDFDocumentIterator()

        for file in files:
            filename = os.path.basename(file)  # Extract the filename from the file path
            for index, (page_meta, content) in enumerate(iterator.iterate(file)):
                # Apply cleaning to the extracted content using the custom cleaner and Unicode cleaner
                cleaned_content = custom_cleaner.modify_document(content)
                cleaned_content = unicode_cleaner.modify_document(cleaned_content)

                # Generate a unique filename for each page using page number and row index
                unique_filename = generate_unique_filename(filename, page_meta['page_number'], cleaned_content, index)

                # Add the cleaned data and metadata to the dataset, including filename
                dataset.append({
                    "text": cleaned_content,
                    "page_number": page_meta["page_number"],
                    "filename": unique_filename  # Ensure unique filename
                })

        # Convert the dataset to a Pandas DataFrame
        df = pd.DataFrame(dataset)

        # Write each row to a separate JSONL file
        for _, row in df.iterrows():
            output_file = os.path.join(output_clean_dir, row['filename'])
            with open(output_file, 'w') as f:
                f.write(row['text'])
        print(f"Finished reformatting {len(files)} files")

    print("Finished reformatting all files")

# Argument parser to handle command-line arguments
def attach_args(
        parser=argparse.ArgumentParser(
            """
    Text cleaning and language filtering
    
    Takes as input a directory consisting of PDF files and outputs a separate
    directory with the extracted and cleaned text. The cleaning removes 
    Unicode characters and special symbols.
    """,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
):
    parser.add_argument(
        "--input-data-dir",
        type=str,
        default=None,
        help="Input directory consisting of PDF files that are accessible "
             "to all nodes. Use this for a distributed file system",
    )
    parser.add_argument(
        "--input-text-field",
        type=str,
        default="text",
        help="The name of the field within each datapoint object of the input "
             "file that contains the text.",
    )
    parser.add_argument(
        "--input-file-type",
        type=str,
        default="pdf",
        help="File type of the dataset to be read in. In this case, 'pdf'.",
    )
    parser.add_argument(
        "--output-clean-dir",
        type=str,
        default=None,
        required=True,
        help="The output directory to where the cleaned JSONL files will be written",
    )
    parser.add_argument(
        "--output-file-type",
        type=str,
        default="jsonl",
        help="File type the dataset will be written to. Supported formats include 'jsonl', 'pickle', or 'parquet'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of files to read into memory at a time.",
    )

    parser = add_distributed_args(parser)
    return parser


# Main script entry point
def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    console_script()
