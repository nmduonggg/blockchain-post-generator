from openai import OpenAI


class GPTWrapper():
    def __init__(self):
        self.client = OpenAI()
        self.MODEL = "gpt-4o"
        self.SYSTEM = """
        You are a helpful assistant that excels in blockchain and bitcoin. You have lots of domain knowledge in blockchain and always support social influencers to grow their audience based on your knowledge.
        """
        self.CONTENT_PATH = './generated_outputs.txt'
        
        
    def get_analyzed_content(self, content_path):
        content = ''
        with open(content_path, 'r') as f:
            for line in f.readlines():
                content += f"{line}\n"
        return content
    
    def wrap_with_default_prompt(self, content):
        default_prompt = """
        Creating engaging and informative content is essential for influencers and marketers to builde credibility and grow their audience. \
            Please summarize the analyzed information and generate an impact ful post based on blockchain data analysis and market trends, to ensure the users paying their attention and grow up influencers' audience.\
                The brief analysis of today's blockchain data analysis are as follows\n"""
        return default_prompt + content
                
        
    def get_message(self):
        message = [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": self.SYSTEM
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": self.wrap_with_default_prompt(self.get_analyzed_content(self.CONTENT_PATH))
                }
            ]
            }
        ]
        return message
        
    def forward(self):
        completion = self.client.chat.completions.create(
            model=self.MODEL,
            messages=self.get_message()
        )
        return completion.choices[0].message
    
if __name__=='__main__':
    gpt_wrapper = GPTWrapper()
    post = gpt_wrapper.forward()
    print(post.content)
        
        
        
        
