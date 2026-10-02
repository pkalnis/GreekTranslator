"""Translate English phrases to Greek using Helsinki-NLP/opus-mt-en-el."""
from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-el"
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
    print(inputs)
    outputs = model.generate(**inputs, max_new_tokens=512, num_beams=1)
    print(outputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    print(f"Loading {MODEL_NAME} ...")
    tokenizer, model = load_model(MODEL_NAME)
    print("Ready. Enter an English phrase (empty line to quit).")

    while True:
        text = input("\nEnglish: ").strip()
        if not text:
            break

        translation = translate(text, tokenizer, model)
        print(f"Greek:   {translation}")


if __name__ == "__main__":
    main()
