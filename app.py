from flask import Flask
from transformers import T5Tokenizer, T5ForConditionalGeneration
import requests
import pandas as pd
app = Flask(__name__)

@app.route('/')
def hello_world():
    api_url = "https://amazon-merchant-data.p.rapidapi.com/get-reviews"
    api_params = {
        "asin": "B0BY2PWDFQ",
        "country": "in",
        "page": "1",
        "reviewerType": "all_reviews",
        "mediaType": "all_contents"
    }
    api_headers = {
        "X-RapidAPI-Key": "1cbf7f20aamsh867b3961a1dfb41p10098ajsn34ac2d88db21",
        "X-RapidAPI-Host": "amazon-merchant-data.p.rapidapi.com"
    }

    response = requests.get(api_url, headers=api_headers, params=api_params)
    data = response.json()

    reviews_data = data.get('reviews', [])
    reviews = [review.get('text', '') for review in reviews_data]


    combined_text = " ".join(reviews)
    summary = generate_summary_t5(combined_text)

    df = pd.DataFrame({"text": reviews})


    with pd.option_context('display.colheader_justify', 'center'):
        print(df.to_string(index=False))

    print("\nOriginal Reviews:\n", "\n".join(reviews))
    print("\nGenerated Summary:\n", summary)


    return "hello world"

def generate_summary_t5(paragraph):
    model_name = "t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer.encode("summarize: " + paragraph, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

