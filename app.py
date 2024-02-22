from flask import Flask
from transformers import T5Tokenizer, T5ForConditionalGeneration
import requests
import pandas as pd
app = Flask(__name__)
@app.route('/')
def hello_world():
    print("again called")
    api_url = "https://amazon-merchant-data.p.rapidapi.com/get-reviews"
    api_params = {
        "asin": "B09W2QF7Q4",
        "country": "in",
        "page": "1",
        "reviewerType": "all_reviews",
        "mediaType": "all_contents"
    }
    api_headers = {
        "X-RapidAPI-Key": "05e9c4569emshc3a68a93bba3031p1a903bjsnc6938a132bcd",
        "X-RapidAPI-Host": "amazon-merchant-data.p.rapidapi.com"
    }
    response = requests.get(api_url, headers=api_headers, params=api_params)
    data = response.json()
    print(data)
    reviews_data = data.get('reviews', [])
    reviews = [review.get('text', '') for review in reviews_data]
    print(reviews)
    # "X-RapidAPI-Key": "2524ada807msh7fad806b55f072ap1153e1jsn226b706c4919",
    combined_text = " ".join(reviews)
    summary = generate_summary_t5(combined_text)
    df = pd.DataFrame({"text": reviews})
    with pd.option_context('display.colheader_justify', 'center'):
        print(df.to_string(index=False))
    print("\nGenerated Summary:\n", summary)
    return summary

def generate_summary_t5(paragraph):
    model_name = "t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name , legacy = False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer.encode("summarize: " + paragraph, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary