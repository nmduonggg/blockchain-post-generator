import gradio as gr
import os
import random
import numpy as np
import argparse
from llm.gpt_wraper import GPTWrapper
from llm.analyser import Agent
import utils
from tqdm import tqdm
import time
import threading

from deepeval.metrics import (AnswerRelevancyMetric, SummarizationMetric,
                              FaithfulnessMetric, ContextualRelevancyMetric)
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval import evaluate


###-- INITIALIZATION --###
analyser = Agent()
generator = GPTWrapper()

MAX_LENGTH = 5000

def open_txt(file):
    with open(file, 'r') as f:
        data = f.read()
    return data

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

def generate_samples():
    
    def setup_option(keyword):    
        analyser.setup_option(keywords_converter[keyword])
        return
    
    def run_analyser(user_prompt, keywords):
        analyser.forward(user_prompt)   # brief analysis written in here
        generation = generator.forward()
        
        with open('./final_post.txt', 'w') as f:
            f.write(generation.content)
        
        return generation.content
    
    def clean_memory():
        global stored_images, analyser
        analyser._clean_memory()
        
    ## start
    samples = []
    keywords_converter = construct_keywords()
    all_keywords = list(keywords_converter.keys())
    for keyword in tqdm(all_keywords, total=len(all_keywords)):
        setup_option(keyword)
        run_analyser("Write a post", keyword)
        
        input_text = open_txt("./generated_outputs.txt")[:MAX_LENGTH]
        output_text = open_txt("./final_post.txt")[:MAX_LENGTH]
        sample = LLMTestCase(
            input = input_text,
            actual_output = output_text,
            retrieval_context = [input_text],
            context = [input_text]
            
        )
        samples.append(sample)
        clean_memory()

        # if len(samples)==3: break
    
    return samples

def main():
    
    summarization_metric = SummarizationMetric(model="gpt-4o", include_reason=False)
    answer_relevancy_metric = AnswerRelevancyMetric(model="gpt-4o", include_reason=False)
    faithfulness_metric = FaithfulnessMetric(model="gpt-4o", include_reason=False)
    contextual_relevancy_metric = ContextualRelevancyMetric(model="gpt-4o", include_reason=False)
    
    test_cases = generate_samples()
    
    dataset = EvaluationDataset(test_cases=test_cases)
    evaluate(
        dataset,
        metrics=[answer_relevancy_metric, summarization_metric, 
                 faithfulness_metric, contextual_relevancy_metric],
        skip_on_missing_params=True,
    )


if __name__=='__main__':
    main()
