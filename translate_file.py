"""Translate an English text file to Greek, writing the result to another file.

Usage: python translate_file.py input.txt output.txt
"""
import argparse
import re
from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"
SAFETENSORS_REVISION = "refs/pr/2"

def load_model(model_name: str):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(
        model_name, revision=SAFETENSORS_REVISION, use_safetensors=True
    )
    model.eval()
    return tokenizer, model


def translate(text: str, tokenizer, model) -> str:
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=512, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
SENTENCE_BOUNDARY = r"(?<=[.!?])\s+"
def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences, e.g. "Hi. How are you?" -> ["Hi.", "How are you?"]."""
    paragraph = paragraph.strip()
    if paragraph:
        return re.split(SENTENCE_BOUNDARY, paragraph)
    else:
        return []


def translate_paragraph(paragraph: str, tokenizer, model) -> str:
    """Translate one paragraph sentence by sentence; an empty paragraph stays empty."""
    sentences = split_sentences(paragraph)
    greek_sentences = []
    for sentence in sentences:
        greek_sentences.append(translate(sentence, tokenizer, model))
    return " ".join(greek_sentences)


def main():
    parser = argparse.ArgumentParser(description="Translate an English text file to Greek.")
    parser.add_argument("input", help="English input text file")
    parser.add_argument("output", help="Greek output text file")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        paragraphs = f.read().split("\n\n")

    print(f"Loading {MODEL_NAME} ...")
    tokenizer, model = load_model(MODEL_NAME)

    print(f"Translating {args.input} ...")
    greek_paragraphs = []
    for i, paragraph in enumerate(paragraphs, start=1):
        greek_paragraphs.append(translate_paragraph(paragraph, tokenizer, model))
        print(f"  paragraph {i}/{len(paragraphs)}")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n\n".join(greek_paragraphs) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
