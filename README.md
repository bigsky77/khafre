# Khafre

Have you ever been lost inside a **MASSIVE** code base or complex system? What if you wanted high-level analysis of the system without getting lost in the weeds?

Inspired by [this post](https://python.langchain.com/en/latest/use_cases/code/twitter-the-algorithm-analysis-deeplake.html?highlight=twitter) on analysis of the Twitter algorithm.  

**Khafre** is designed to take any arbitrary-sized codebase alongside questions about that codebase and generate a simple and high-level overview of those answers. Even better, the system is designed to post those answers directly to Twitter in the form of full tweet threads.  Our next step is to incorporate Telegram and Discord chat logs.  This will enable question answering that reviews context from chats, code, and docs! 

# How to Use<a name="how-to-use"></a>
To use the script, you will need to follow these steps:

1. Clone the repository via `git clone https://github.com/bigsky77/khafre.git` and `cd` into the cloned repository.
2. Install the required packages: `pip install -r requirements.txt`
3. Copy the .env.example file to .env: `cp .env.example .env`. This is where you will set the following variables.
4. Set your OpenAI, DeepLake, and Twitter API keys in your new .env file
4. Run the script. `python runner.py`
