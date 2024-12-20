import os
import json
from datetime import datetime
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
from transformers.image_utils import load_image

from openai import OpenAI

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def to_datetime(timestamp):
    dt_object = datetime.utcfromtimestamp(timestamp)
    return dt_object.strftime('%Y-%m-%d %H:%M').replace(' ', '\n')

def write2file(content, fp, rewrite=False):
    if rewrite:
        with open(fp, 'w') as f:
            f.write('The analysis of blockchain-related requests from users.\n' + content +'\n')    
    else:
        with open(fp, 'a') as f:
            f.write(content + '\n')
    return

class Agent():
    def __init__(self, root='./analysis', out_file = './generated_outputs.txt', initialize=True):
        super().__init__()
        self.processor, self.model = None, None
        self.PROJECT_OPT_PATH = './analysis/projects/keywords/profile_list.json'
        
        
        if initialize:
            self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Synthetic")
            self.model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Synthetic",
                                                torch_dtype=torch.float16).to(DEVICE)

            print("Load model done")
        
        self.root=root
        self.out_file = out_file
        self.default_options = ['bnb', 'eth', 'projects']
        with open(self.PROJECT_OPT_PATH, 'r') as f:
            self.projects_list = json.load(f)
        self.project_options = [pj['name'] for pj in self.projects_list]
        self.all_options = self.default_options + self.project_options
        
        self.choosen_images = []
        
        self._clean_memory()    # re-initalize
        
    def _clean_memory(self):
        write2file('-----', self.out_file, True)
        
    def _craft_prompt(self, target):
        prompt = f"""
        The following is a plot that shows {target} across continous timestamps.
        The timestamps in x-axis are of the form Year-Month-Day Hour-Minute if it is line or scatter plot, and categories if it is a bar graph.
        Statistically analyze and summarize it by describing the trend; depicting outliers if any; showing interesting observations if applicable and so on. 
        Provide numerical evidences but not the values of x-axis and y-axis.
        Each of your statements should be different, not repeated.
        Remove gramatical but redundant phrases such as "there is" or "the plot shows", etc.
        The value can be written in the scientific forms for simplicity. 
        Summarize each observation or analysis as a short meaningful phrase.
        """
        return prompt
    
    def _craft_prompt_insight(self, target):
        prompt = f"""
        Do not hallucinate. Do not make up information.
        The following is a plot that shows {target} across continous timestamps.
        The timestamps in x-axis are of the form Year-Month-Day Hour-Minute or Timestamp with respect to January 1, 1970.
        Statistically analyze the plot via describing the trend; depicting outliers if any; showing interesting observations if applicable.
        Provide numerical evidences but not the values of x-axis and y-axis. The output texts should be a set of keywords describing interesting features from the plots.
        Do not repeat the request of users.
        """
        return prompt
    
    def reinit(self):
        with open(self.PROJECT_OPT_PATH, 'r') as f:
            self.projects_list = json.load(f)
        self.project_options = [pj['name'] for pj in self.projects_list]
        self.all_options = self.default_options + self.project_options
        
        
        
        
    def get_all_options(self):
        return self.all_options
        
    def setup_option(self, option):
        assert option in self.all_options, 'inavailable option'
        
        self.option = option
        choosen_folder = os.path.join(self.root, option)
        
        if not os.path.isdir(choosen_folder):
            chosen_project = self.projects_list[self.project_options.index(option)]
            assert chosen_project['name'] == option
            choosen_folder = chosen_project['path']
            self.chosen_project = chosen_project
            
        images = list()
        prompt_keywords = list()
        for fn in os.listdir(choosen_folder):
            if not fn.endswith(('.png', '.jpg')): continue
            fp = os.path.join(choosen_folder, fn)
            images.append(load_image(fp))
            
            # keyword handler
            compressed_keyword = fn.split('.')[0]
            keyword = compressed_keyword.replace('_', ' ')
            prompt_keywords.append(keyword)
        
        self.choosen_images = images
        self.prompt_keywords = prompt_keywords
        self.full_prompts = [self._craft_prompt(kw) for kw in self.prompt_keywords]
        return images, prompt_keywords

    # Create input messages
    def create_message(self, image, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            },
        ]

        # Prepare inputs
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(DEVICE)
        
        return inputs

    # Generate outputs
    def vlm_reply(self, inputs, max_tokens=200):
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return generated_texts
    
    def forward(self, user_prompt=None):
        assert hasattr(self, 'choosen_images'), 'option goes first'
        
        full_content = ""
        for image, prompt, kw in zip(self.choosen_images, self.full_prompts, self.prompt_keywords):
            
            inputs = self.create_message(image, prompt)
            generated_text = self.vlm_reply(inputs)[0]
            
            start_idx = generated_text.find('Assistant: ')
            generated_text = generated_text[start_idx + len('Assistant: '):]
            generated_text = generated_text.replace('\n', ' ')

            content = f"""{kw.capitalize()} analysis: {generated_text}"""
            if self.option in self.project_options:
                stat_content = ''
                for k, v in self.chosen_project.items():
                    if k not in ['timestamps', 'path']:
                        stat_content += f"{k}: {v}\n"
                    if 'predicted' in k: 
                        predictions = list(self.chosen_project[k])
                        for pts, pvl in zip(self.chosen_project['predict_timestamps'], predictions):
                            stat_content += f"{k.replace('_', ' ')} for {to_datetime(pts)} is {pvl}. "
                content += "\n" + stat_content
            full_content += content + '\n'
                   
        if user_prompt is not None: full_content = user_prompt + "\n-----\n" + full_content
        write2file(full_content, self.out_file)
            
        
            
class GPTAgent(Agent):
    def __init__(self, root='./analysis', out_file = './generated_outputs.txt'):
        super().__init__(root, out_file, initialize=False)
        self.client = OpenAI()
        self.MODEL = "gpt-4o-mini"
        self.SYSTEM = """
        You are a helpful assistant that excels in blockchain and bitcoin. You have lots of domain knowledge in blockchain and always support social influencers to grow their audience based on your knowledge.
        """
        PROJECT_OPT_PATH = './analysis/projects/keywords/profile_list.json'
        with open(PROJECT_OPT_PATH, 'r') as f:
            self.projects_list = json.load(f)
        self.project_options = [pj['name'] for pj in self.projects_list]
        self.default_options = self.all_options
        self.all_options = self.default_options + self.project_options
        
    def get_all_options(self):
        return self.all_options
        
    def setup_option(self, option):
        assert option in self.all_options, 'inavailable option'
        
        self.option = option
        choosen_folder = os.path.join(self.root, option)
        
        if not os.path.isdir(choosen_folder):
            chosen_project = self.projects_list[self.project_options.index(option)]
            assert chosen_project['name'] == option
            choosen_folder = chosen_project['path']
            
        images = list()
        prompt_keywords = list()
        for fn in os.listdir(choosen_folder):
            if not fn.endswith(('.png', '.jpg')): continue
            fp = os.path.join(choosen_folder, fn)
            images.append(load_image(fp))
            
            # keyword handler
            compressed_keyword = fn.split('.')[0]
            keyword = compressed_keyword.replace('_', ' ')
            prompt_keywords.append(keyword)
        
        self.choosen_images = images
        self.prompt_keywords = prompt_keywords
        self.full_prompts = [self._craft_prompt(kw) for kw in self.prompt_keywords]
        return images, prompt_keywords

        
    def create_message(self, images, prompts):
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "image": image
                    },
                ],
            }
        for image, prompt in zip(images, prompts)]
        
        return messages
        
    def forward(self):
        assert hasattr(self, 'choosen_images'), 'option goes first'
        
        messages = self.create_message(self.choosen_images, self.full_prompts)
        completion = self.client.chat.completions.create(
            model=self.MODEL,
            messages=messages
        )
        for choice in completion.choices:
            generated_text = choice.message
            # start_idx = generated_text.find('Assistant: ')
            # generated_text = generated_text[start_idx + len('Assistant: '):]
            generated_text = generated_text.replace('\n', ' ')

            # content = f"""{kw.capitalize()} analysis: {generated_text}"""
            content = generated_text
            write2file(content, self.out_file)
            
            
if __name__=='__main__':
    agent = Agent()
    all_options = agent.get_all_options()
    print(all_options)
    agent.setup_option('AGIKing')
    agent.forward()