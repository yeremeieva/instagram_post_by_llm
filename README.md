# Instagram Post Generator
## Overview
This Python script generates Instagram shop posts based on the user input (post content and additional parameters) and predefined templates. It utilizes Gradio for the user interface, Langchain to proccess the query and Hugging Face model CohereForAI/c4ai-command-r-plus for natural language processing and text generation.

## Features
* Allows users to input post content and specify parameters such as type, target audience, tone, instagram username etc.
* Generates Instagram posts based on the provided input and predefined template with similar examples from an existing Instagram page.
* Provides a user-friendly interface for generating Instagram posts.

## Structure
Project contains two main files: **logic.ipynb** and **gradio_ui.py**
### logic.ipynb
This file explains main logic of the project. We start with scraping the Instagram page and then extract the data from it. This data will be used to provide language model with examples. Then we create a prompt template with context and rules how to write a post. We use similarity search to include only similar to our query (post content) examples into the prompt. This prompt goes into the LLM and returns required post.

### gradio_ui.py 
We use the logic from the previous file to create gradio interface. User can specify the following parameters:
* Post content
* Common or custom type of a post
* Common or custom target audience
* Common or custom tone of voice
* Instagram shop name
* Add or do not add hashtags and emoji
* Change LLM parameters (number of tokens and temperature)

## How to Use
### Installation: 

Install the required packages listed in requirements.txt.

```bash
pip install -r requirements.txt
```
Provide your Hugging Face API Token

```bash
os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''
```

Run the Script: Execute the script gradio_ui.py.

```bash
python gradio_ui.py
```

### Use the Interface:
* Use the UI localy or share the link. Input your post content and specify parameters for the post.
* Generate Post: Click the "Generate" button to generate your Instagram post.
* Modify parameters and generate more!
