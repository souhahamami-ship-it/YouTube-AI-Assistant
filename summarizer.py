from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


def load_model():

    model_name = "facebook/bart-large-cnn"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    return tokenizer, model


def summarize_text(text, tokenizer, model):

    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        do_sample=False
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary


def summarize_transcript(
    transcript,
    tokenizer,
    model,
    progress_callback=None
):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(transcript)

    chunk_summaries = []

    for i, chunk in enumerate(chunks):

        summary = summarize_text(
            chunk,
            tokenizer,
            model
        )

        chunk_summaries.append(summary)

        if progress_callback:
            progress_callback(
                (i + 1) / len(chunks)
            )

    combined_summary = " ".join(
        chunk_summaries
    )

    final_summary = summarize_text(
        combined_summary,
        tokenizer,
        model
    )

    return final_summary, len(chunks)