import gradio as gr
import pandas as pd
import os
import re
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector


def read_and_clean_dataset() -> list:
    dataset = pd.read_csv('data/dataset.csv')
    partial_dataset = dataset.loc[:, ['caption', 'likesCount', 'ownerUsername']]

    partial_dataset_ospreypacks = partial_dataset[partial_dataset['ownerUsername'] == 'ospreypacks']
    partial_dataset_liked = partial_dataset_ospreypacks[partial_dataset_ospreypacks['likesCount'] > 0]

    sorted_dataset = partial_dataset_liked.sort_values(by='likesCount', ascending=False).reset_index(drop=True)
    top_50_most_liked_posts = sorted_dataset.head(50)

    return list(top_50_most_liked_posts['caption'])


def example_template() -> PromptTemplate:
    template = "Instagram Post: {post}"
    example_prompt = PromptTemplate(template=template, input_variables=['post'])

    return example_prompt


def example_selector(captions: list) -> SemanticSimilarityExampleSelector:
    examples = [{'post': caption} for caption in captions]

    selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        HuggingFaceEmbeddings(),
        Chroma,
        k=3
    )
    return selector


def prompt_template() -> FewShotPromptTemplate:
    selector = example_selector(read_and_clean_dataset())

    prefix = """ Instagram Post Instructions for LLM Model

    ## 1. Overview
    You are a top-tier algorithm designed for creating modern Instagram posts for an online shop. You will get a query with an explanation of the post and additional parameters:
     * Type of post;
     * Target audience;
     * Tone and style;
     * Name of Instagram shop;
     * Additional requirements or comments;

    ## 2. Type of post
    Post can be promotional (product description) or advertisement (promotion, sales, giveaway)or informational (tips, news, event information).
    Sometimes shop can ask for another style. Strictly follow the type of the post required.

    ## 3. Target audience
    Target audience can be Mountaineers, Travelers, Active athletes or Other customers. Please use your knowledge to better sell shop's product for this audience.

    ## 4. Tone and style
    The tone of the post can be Professional or Casual or Motivational or Other. Strictly follow the tone of the post required.

    ## 5. Name of Instagram shop
    It is absolutely important to use proper shop name. In the examples section, you will see #OspreyPacks. Change it with the provided shop name in the hashtags if they are required.

    ## 6. Additional requirements or comments
    Shop can ask for presence of hashtags, request for specific vocabulary or phrases, or some other comments.
    The use of hashtags is 5 words. You are not allowed to use more than 5 hashtags.

    ## 7. Strict Compliance
    Adhere to the rules strictly. Non-compliance will result in termination.
    
    ## 8. Not correct input
    If the input is not correct than write "Your input is not correct, Please try again."
    Examples of incorrect input: 123, 'hi!', write some code, etc. 
    You are allowed only to generate instagram posts.

    Some examples of good posts from mountain backpacks are here:

    """

    suffix = """
    ---
    Shop query will include post content and additional parameters: type of post, target audience, tone, 
    name of Instagram shop,  presence of hashtags and additional comments.
    Please create required and beautiful post from content for this shop. Follow the Instructions. 

    Shop query: {query}

    ---
    Instagram Post: """

    dynamic_prompt_template = FewShotPromptTemplate(
        example_selector=selector,
        example_prompt=example_template(),
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n"
    )

    return dynamic_prompt_template


def init_llm(number_of_tokens: int, temperature: float, prompt: FewShotPromptTemplate) -> LLMChain:
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''
    model = HuggingFaceEndpoint(repo_id='CohereForAI/c4ai-command-r-plus', temperature=temperature,
                                max_new_tokens=number_of_tokens, stop_sequences=['---'])

    llm_chain = LLMChain(prompt=prompt, llm=model)
    return llm_chain


def check_other(main, other):
    if main is None:
        return other
    else:
        return main


def check_hashtag_emoji(main):
    if main is True:
        return 'add'
    else:
        return 'no'


def run_query(post_content, type_of_post, type_other, target_audience, audience_other, tone, tone_other, name_of_shop,
              hashtags, emoji, tokens, temperature):
    template = prompt_template()
    llm_chain = init_llm(tokens, temperature, template)

    example_query = f"""Post content: {post_content},
    Type of post: {check_other(type_of_post, type_other)},
    Target Audience: {check_other(target_audience, audience_other)},
    Tone and Style: {check_other(tone, tone_other)},
    Name of Instagram shop: {name_of_shop},
    Additional requirements: {check_hashtag_emoji(hashtags)} hashtags, {check_hashtag_emoji(emoji)} emoji. """

    result = llm_chain.run(example_query)
    words = re.split('[\n ]', result)

    return result, len(words)


def gradio_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="Instagram Post Generator") as demo:
        gr.Markdown(
            """
        # Generate your Instagram shop post!
        Start typing your post content below.
        """
        )
        with gr.Row():
            post_content = gr.Textbox(lines=3, label="Post Content")

        gr.Markdown("Specify few parameters of the post you want.")
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                type_of_post = gr.Radio(choices=["Promotional (product description)", "Advertisement (sales, giveaway)",
                                        "Informational (tips, news, event, information)"],
                                        label="Common Type of Post")
            with gr.Column(scale=2, min_width=600):
                type_other = gr.Textbox(label="Custom Type of Post", visible=True, lines=2)

        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                target_audience = gr.Radio(choices=["Mountaineers", "Travelers", "Active athletes", "Climbers"],
                                       label="Common Target Audience")
            with gr.Column(scale=2, min_width=600):
                audience_other = gr.Textbox(label="Custom Target Audience")

        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                tone = gr.Radio(choices=["Professional", "Casual", "Motivational"], label="Common Tones of Voice")
            with gr.Column(scale=2, min_width=600):
                tone_other = gr.Textbox(label="Custom Tones of Voice", visible=True, lines=1)

        gr.Markdown("Add the username of your Instagram page.")
        with gr.Row():
            name_of_shop = gr.Textbox(label="Name of Instagram shop", visible=True, lines=1)

        gr.Markdown("Add hashtags or emoji.")
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                hashtags = gr.Checkbox(label="Add Hashtags", visible=True)
            with gr.Column(scale=1, min_width=600):
                emoji = gr.Checkbox(label="Add Emoji", visible=True)

        gr.Markdown("Modify parameters of the Language Model.")
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                tokens = gr.Number(label="Max number of tokens", visible=True, value=100)
            with gr.Column(scale=2, min_width=600):
                temperature = gr.Slider(label="Creativeness of model", visible=True,
                                        minimum=0.01, maximum=0.99, step=0.01, value=0.5)

        with gr.Row():
            btn = gr.Button("Generate")
            clear_btn = gr.ClearButton([post_content, type_of_post, type_other, target_audience, audience_other, tone,
                                        tone_other, name_of_shop, hashtags, emoji, tokens, temperature, ])

        gr.Markdown("### Here is your post!!")
        with gr.Row():
            with gr.Column(scale=1, min_width=800):
                output_text = gr.Textbox(label="Generated Instagram Post")
            with gr.Column(scale=1, min_width=200):
                result_tokens = gr.Number(label="Number of tokens")

        btn.click(run_query, inputs=[post_content, type_of_post, type_other, target_audience, audience_other, tone,
                                     tone_other, name_of_shop, hashtags, emoji, tokens, temperature],
                  outputs=[output_text, result_tokens])

    demo.launch(share=True)


if __name__ == '__main__':
    gradio_ui()
