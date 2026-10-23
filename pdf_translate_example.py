"""Translate an English pdf file to Greek, writing the result to another file.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# re           : Python's built-in regular-expression module, used below to
#                split paragraphs into sentences.
# pymupdf      : the PyMuPDF library (pip install pymupdf), used to open the
#                PDF and extract its text. (Older code imports it as "fitz".)
# transformers : Hugging Face's library of pretrained neural-network models
#                (pip install transformers torch sentencepiece).
#                MarianMTModel is the translation model itself.
#                MarianTokenizer converts text into the numbers (token IDs)
#                the model works with, and converts the model's output back
#                into text.
import re
import pymupdf
from transformers import MarianMTModel, MarianTokenizer

# Name of the pretrained model on the Hugging Face Hub (huggingface.co).
# The "opus-mt" models were trained by the University of Helsinki on OPUS,
# a large collection of texts with their translations. "en-el" means
# English -> Greek ("el" is the language code for Greek, from "Ellinika").
MODEL_NAME = "Helsinki-NLP/opus-mt-en-el"
# Which version of the model files to download. A model on the Hub is stored
# like a git repository; "refs/pr/2" is pull request #2 of that repository,
# which provides the model weights in the "safetensors" format (a safe and
# fast file format for model weights).
SAFETENSORS_REVISION = "refs/pr/2"


# ---------------------------------------------------------------------------
# Load the model
# ---------------------------------------------------------------------------
# Downloads the tokenizer and the model (only the first time; afterwards they
# are cached on disk) and loads them into memory. Both are returned together
# as a tuple, so the caller can write:  tokenizer, model = load_model(...)
def load_model(model_name: str):
    # The tokenizer splits text into small pieces (words or parts of words)
    # and maps each piece to an integer ID from the model's vocabulary.
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    # The model is the neural network that turns English token IDs into
    # Greek token IDs. revision= selects the version of the files (see
    # SAFETENSORS_REVISION above); use_safetensors=True asks for the
    # .safetensors weight file.
    model = MarianMTModel.from_pretrained(
        model_name, revision=SAFETENSORS_REVISION, use_safetensors=True
    )
    # Put the model in "evaluation" (inference) mode, which switches off
    # behaviour that is only wanted during training, such as dropout.
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Translate one piece of text
# ---------------------------------------------------------------------------
# Using a translation model always takes three steps:
#   1. encode  : text      -> token IDs   (done by the tokenizer)
#   2. generate: token IDs -> token IDs   (done by the model)
#   3. decode  : token IDs -> text        (done by the tokenizer)
def translate(text: str, tokenizer, model) -> str:
    # Step 1. The tokenizer works on a list (a "batch") of texts, so we wrap
    # our single text in [ ]. The options mean:
    #   return_tensors="pt" : return PyTorch tensors, the input type the
    #                         model needs
    #   padding=True        : pad shorter texts so that all texts in the
    #                         batch have the same length (has no effect here,
    #                         because the batch contains only one text)
    #   truncation=True     : cut off any text longer than the model's
    #                         maximum input length (512 tokens). This is one
    #                         reason the program translates sentence by
    #                         sentence instead of whole pages at once.
    # The result behaves like a dictionary with the keys "input_ids" (the
    # token IDs) and "attention_mask" (which positions are real text and
    # which are padding).
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    # Step 2. Generate the Greek token IDs. "**inputs" unpacks the dictionary
    # into keyword arguments, i.e. generate(input_ids=..., attention_mask=...).
    #   max_new_tokens=512 : stop after at most 512 output tokens
    #   num_beams=4        : "beam search": at each step, keep the 4 most
    #                        promising partial translations instead of only
    #                        the single best next word. This is slower but
    #                        usually gives better translations.
    outputs = model.generate(**inputs, max_new_tokens=512, num_beams=4)
    # Step 3. outputs contains one translated sequence per input text. We
    # take the first (and only) one, outputs[0], and turn it back into a
    # string. skip_special_tokens=True removes internal markers such as
    # <pad> (padding) and </s> (end of sentence).
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------------------------------------------------------
# Split a paragraph into sentences
# ---------------------------------------------------------------------------
# Regex that matches the gap between two sentences, used with re.split().
#
#   (?<=[.!?])  Lookbehind: the match must be immediately preceded by a period,
#               exclamation mark or question mark. A lookbehind only checks the
#               preceding character; it does not consume it, so the punctuation
#               is not part of the match and stays attached to its sentence.
#               [.!?] is a character class: any one of the three characters
#               (inside [], "." is a literal period, not "any character").
#   \s+         One or more whitespace characters (spaces, tabs). This is the
#               part actually matched, so it is what re.split() removes.
#
# Example: "Hi. How are you? Fine" -> the regex matches the space after "Hi."
# and the space after "you?", so re.split() gives ["Hi.", "How are you?", "Fine"].
# Punctuation not followed by whitespace (e.g. the "." in "3.14") is not a match.
#
# The "r" before the string makes it a raw string: the backslash in "\s" is
# passed to the regex engine as-is instead of being treated as a Python
# escape sequence.
SENTENCE_BOUNDARY = r"(?<=[.!?])\s+"
def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences, e.g. "Hi. How are you?" -> ["Hi.", "How are you?"]."""
    # strip() removes whitespace (spaces, tabs, newlines) from both ends.
    paragraph = paragraph.strip()
    # In an if-statement a non-empty string counts as True and an empty
    # string as False, so this tests "is there any text left?".
    if paragraph:
        # re.split() cuts the string at every place where SENTENCE_BOUNDARY
        # matches and returns the pieces as a list.
        # Limitation: this simple rule also splits after abbreviations,
        # e.g. "Dr. Smith" becomes ["Dr.", "Smith"].
        return re.split(SENTENCE_BOUNDARY, paragraph)
    else:
        # No text: return an empty list. (re.split() on an empty string would
        # return [""], a list containing one empty sentence, which we do not
        # want to send to the model.)
        return []


# ---------------------------------------------------------------------------
# Translate a whole paragraph
# ---------------------------------------------------------------------------
# Why sentence by sentence? The model was trained on pairs of single
# sentences, so it translates single sentences best, and short inputs never
# reach the 512-token limit mentioned in translate().
def translate_paragraph(paragraph: str, tokenizer, model) -> str:
    """Translate one paragraph sentence by sentence; an empty paragraph stays empty."""
    sentences = split_sentences(paragraph)
    # Translate each sentence and collect the Greek results in a list.
    greek_sentences = []
    for sentence in sentences:
        greek_sentences.append(translate(sentence, tokenizer, model))
    # " ".join(list) glues the strings of the list together, with one space
    # between them. For an empty list it returns "", so an empty paragraph
    # stays empty.
    return " ".join(greek_sentences)



# ---------------------------------------------------------------------------
# Read the text of the PDF
# ---------------------------------------------------------------------------
# Returns all paragraphs of the PDF, in reading order, as a list of strings.
# fname is the name (path) of the PDF file.
def read_pdf_file(fname) -> list[str]:
    # Open the PDF. The document object behaves like a list of pages.
    doc = pymupdf.open(fname)

    paragraphs = []
    # Loop over every page of the document ...
    for page in doc:
        # ... and over every text block on that page. get_text("blocks")
        # groups the text of a page into rectangular blocks (usually a
        # paragraph, a heading, a column, ...). Each block is a tuple:
        #   (x0, y0, x1, y1, text, block_no, block_type)
        # x0, y0, x1, y1 give the block's position on the page.
        for block in page.get_text("blocks"):
            # block[4] is the text of the block.
            # Inside a block, lines are separated by "\n". The sequence
            # "\n \n" (a line that contains only a space) is an empty line,
            # which we treat as a break between two paragraphs. split()
            # cuts the text at every such break and returns a list of strings.
            block_text = block[4].split("\n \n") # the 4th element is the text
            for paragraph in block_text:
                # Remove whitespace from both ends and skip empty pieces.
                paragraph = paragraph.strip()
                if paragraph:
                    paragraphs.append(paragraph)
    return paragraphs


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main():
    # Input and output file names. These are relative paths: the PDF must be
    # in the folder you run the program from, and output.txt is created there.
    infile = "document.pdf"
    outfile = "output.txt"

    # 1. Extract the English paragraphs from the PDF.
    paragraphs = read_pdf_file(infile)

    # 2. Load the translation model. The first run downloads it from the
    #    internet (a few hundred MB), so it may take a while.
    #    f"..." is an f-string: an expression inside { } is replaced by its
    #    value, so this prints "Loading Helsinki-NLP/opus-mt-en-el ...".
    print(f"Loading {MODEL_NAME} ...")
    tokenizer, model = load_model(MODEL_NAME)

    # 3. Translate every paragraph, printing the progress as we go.
    print(f"Translating {infile} ...")
    greek_paragraphs = []
    # enumerate() gives a counter together with each item of the list.
    # start=1 makes the counter begin at 1 instead of 0, so the progress
    # message reads "paragraph 1/20" rather than "paragraph 0/20".
    for i, paragraph in enumerate(paragraphs, start=1):
        greek_paragraphs.append(translate_paragraph(paragraph, tokenizer, model))
        print(f"  paragraph {i}/{len(paragraphs)}")

    # 4. Write the translation to the output file.
    #    "w" opens the file for writing (any old content is replaced).
    #    encoding="utf-8" is needed to store Greek letters correctly; the
    #    default encoding on some systems (e.g. Windows) cannot store them.
    #    The with-statement closes the file automatically when the block
    #    ends, even if an error occurs.
    #    "\n\n".join(...) puts an empty line between paragraphs, and the
    #    final "\n" ends the file with a newline.
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\n\n".join(greek_paragraphs) + "\n")
    print(f"Wrote {outfile}")


# __name__ is "__main__" only when this file is run directly
# (python pdf_translate_example.py). When another program imports this file
# to reuse its functions, __name__ is the module name instead, so main() is
# not called and no translation starts.
if __name__ == "__main__":
    main()