import gradio as gr
import os
import random
import numpy as np
import argparse
from llm.gpt_wraper import GPTWrapper
from llm.analyser import Agent
import utils
import time
import threading


###-- INITIALIZATION --###

analyser = Agent()
generator = GPTWrapper()

#-- construct insight keywords
def construct_keywords():
    global analyser, generator
    analyser.reinit()
    
    keywords_converter = {
        'summary blockchain': 'bnb',
        'summary ethereum': 'eth',
        'summary projects': 'projects'}

    #--arima insights
    projects_list = analyser.projects_list
    projects_options = analyser.project_options
    for i, opt in enumerate(projects_options):
        project = projects_list[i]
        
        p = random.random()
        
        if p < 0.25:
            prop = random.choice(['trend', 'mean', 'max', 'min'])
            keywords_converter[f"{opt} past {prop} is {round(project[prop], 2) if prop != 'trend' else project[prop]}"] = opt
        
        ##-- arima keyword
        predicted_key = 'predicted_' + project['value_type']
        peak_value = np.max(project[predicted_key])
        peak_timestamp = project['predict_timestamps'][int(project[predicted_key].index(peak_value))]
        peak_date = utils.to_datetime(peak_timestamp).replace('\n', '-')
        peak_value = round(peak_value / 1e6, 2)
        
        if (0.25 <= p) and (p < 0.5):
            keywords_converter[f"{opt} expectedly get peak at {peak_value}M on {peak_date}"] = opt
        
        min_value = np.min(project[predicted_key])
        min_timestamp = project['predict_timestamps'][int(project[predicted_key].index(min_value))]
        min_date = utils.to_datetime(min_timestamp).replace('\n', '-')
        min_value = round(min_value / 1e6, 2)
        
        if (0.5 <= p) and (p < 0.75):
            keywords_converter[f"{opt} expectedly drop to {min_value}M on {min_date}"] = opt

        if p>=0.75:
            if min_timestamp > peak_timestamp:
                if (min_value - peak_value >=0.1):
                    keywords_converter[f"{opt} expectedly raise to {peak_value}M on {peak_date} before drop to {min_value}M on {min_date}"] = opt
                else:
                    keywords_converter[f"{opt} expectedly reach to {min_value}M on {min_date} after slight fluctuation"] = opt
                    
            elif (peak_timestamp > min_timestamp):
                if (peak_value - min_value >=0.1):
                    keywords_converter[f"{opt} expectedly drop to {min_value}M on {min_date} before experiences a increase by {round(peak_value - min_value, 2)}M on {peak_date}"] = opt
                else:
                    keywords_converter[f"{opt} expectedly reach to {peak_value}M on {peak_date} after slight fluctuation"] = opt
                    
    return keywords_converter

keywords_converter = construct_keywords()
##-------------------------##

# Define a function to generate text based on selected keyword
def setup_option(keyword):    
    analyser.setup_option(keywords_converter[keyword])
    return

def run_analyser(user_prompt, keywords):
    analyser.forward(user_prompt)
    generation = generator.forward()
    return generation.content, generation.content

# Function to load images from a local directory
stored_images = list()
def load_images(keyword):
    global stored_images
    images, _ = analyser.setup_option(keywords_converter[keyword])
    stored_images += images
    return stored_images

def clean_memory():
    global stored_images
    analyser._clean_memory()
    stored_images = list()
    image_gallery = gr.Gallery(label="Visualization", columns=3, object_fit="contain", height="auto")  # Display images in 3 columns
    user_prompt = gr.Textbox(label="Type your prompt here", interactive=True)
    generated_text = gr.Textbox(label="Generated Text", interactive=False, lines=5)
    
    return image_gallery, user_prompt, generated_text
    
# Auto refresh on the background
def background_process(keywords):
    global keywords_converter
    while True:
        # querying data
        os.system("python ./preprocess/crawl.py")
        # preprocess
        os.system("python ./preprocess/processing.py")
        
        keywords_converter = construct_keywords()
        
        time.sleep(100)  # Simulate a background process

def update_keywords(keywords):
    global keywords_converter
    return gr.Radio(keywords_converter.keys(), label="Select a Keyword")

def ui_launch(launch_kwargs):
    
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            ## Post Recommendation for Social Influencers - Blockchain Lovers
            ### Instructions:
            - Click into keywords to generate post based on our analysis on it.
            - The information and keywords are added incrementally leading to longer inference as the number of keywords grows up.
            - To avoid it, please click "Clean Memory" button to reset.       
            - To reset the insights, click "Refresh Insights". 
            """)
        
        with gr.Row():  # Left column with keyword options
            with gr.Column():
                reset = gr.Button("Clean Memory")
                refresh = gr.Button("Refresh Insights")
                keywords = gr.Radio(keywords_converter.keys(), label="Select a Keyword")
            with gr.Column():  # Right column with generated text output
                user_prompt = gr.Textbox(label="Type your prompt here", interactive=True)
                generated_text = gr.Textbox(label="Generated Text", interactive=False, lines=5)
                generate = gr.Button("Generate Post")
        
        with gr.Row():  # Lower region to display images
            image_gallery = gr.Gallery(label="Visualization", columns=3, object_fit="contain", height="auto")  # Display images in 3 columns
        with gr.Row():
            markdown = gr.Markdown(
                label="Generated Text", show_copy_button=True)
        
        reset.click(clean_memory, inputs=[], outputs=[image_gallery, user_prompt, generated_text])
        refresh.click(update_keywords, inputs=[], outputs=[keywords])
        
        keywords.change(setup_option, inputs=keywords, outputs=[])
        keywords.change(load_images, inputs=keywords, outputs=image_gallery)
        
        generate.click(run_analyser, inputs=[user_prompt, keywords], outputs=[generated_text, markdown])
        
        # Start background task in a separate thread
        thread = threading.Thread(target=background_process, args=[keywords, ], daemon=True)
        thread.start()
        
        demo.launch(**launch_kwargs)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser
    if args.share:
        launch_kwargs['share'] = args.share

    ui_launch(launch_kwargs)




