# Blockchain Post Generator

### This is a demonstration for project of course "Intro to Business Analysis", Hanoi University of Science and Technology, Hanoi, Vietnam.

**__Note__**

Due to limited resources, we hereby crawl and process only a small amount of data (50K objects from database). The output analysis might be inaccurate in some rare scenarios. However, this does not affect the performance of the system and we firmly believe it can be scalable in larger server or real-time production environments.

## Steps to run

1. Define OpenAI's API key for ChatGPT initialization
```
export OPENAI_API_KEY=...
```

2. Initialize the data in the first running. 

- Crawling, preprocessing data. This step will be automated after the first try.
```
python preprocess/crawl.py
python preprocess/processing.py
```

3. Deploy application
- For local deployment
```
python main_app.py
```
- For public and shareable link (3 days)
```
python main_app.py --inbrowser --share
```