#!/usr/bin/env python3
import os
import openai
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

load_dotenv()

# set the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
embeddings = OpenAIEmbeddings()

def agent():
    # load the dataset
    db = DeepLake(dataset_path="hub://davitbun/twitter-algorithm", read_only=True, embedding_function=embeddings)

    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 20

    # Load model
    model = ChatOpenAI(model_name='gpt-3.5-turbo') # 'gpt-3.5-turbo',
    qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)

    questions = [
        "Describe at a high level how the ranking system works",
        "What are three core components of the ranking system?",
    ]
    chat_history = []

    for question in questions:
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")

    # Content summary
    summary_prompt = PromptTemplate(
            input_variables=["chat_history", "questions"],
            template=""" You are an agent with the task of creating an exciting short essay.  Your goal is to make the content engaging and accessible.
            You will recieve a list of {questions} and answers {chat_history}.  All the questions are related and cover the same topic.
            Your task it to write a short essay that summarizes the topics.  You can use the questions and answers to help you write the essay.

            Each paragraph should consist or two to three complete sentences.
            Your essay should be exactly five paragraphs long.

            The first paragraph should be an introduction that cleary states the topic that will be covered and why it matters to the reader.
            The second, third, and fourth paragraphs should each explain one key element of the topic.
            The fifth paragraph should be a conclusion that summarizes the main points of the essay.

            Do not refer to yourself.  Use the metaphor of a marketplace, tweets are the goods being sold and algorithm evaluates who sees them.
            The essay should be written in the present tense and should be written in the active voice.

            At the end of each paragraph add a "/" and the paragraph number.  For example, "/1"
            """
        )
    review_prompt = PromptTemplate(
        input_variables=["thread"],
        template="""You are a content review and consolidation agent.  You will recieve a list of paragraphs {thread}.
        Ensure that each paragraph in the list has no more than 140 character. If a paragraph exceeds 140 characters, split it into two or more smaller paragraphs.
        If a paragraph is longer than 140 characters, you should split it into two or more paragraphs.
        Every paragraph should end with a "/" and the paragraph number. For example, "/1".  If you split a paragraph make sure to update the paragraph number.

        Remember, the maximum length of each paragraph must be less than 140 characters(letters).  DO NOT INCLUDE A PARAGRAPH WITH MORE THAN 140 CHARACTERS.
        Make sure that the paragraphs are in the correct order and that their are no repeat numbers.

        """
        )

    summary_chain = LLMChain(llm=model, prompt=summary_prompt, output_key="thread")
    review_chain = LLMChain(llm=model, prompt=review_prompt, output_key="review")

    overall_chain = SequentialChain(chains=[summary_chain, review_chain], input_variables=["chat_history", "questions"], output_variables=["thread", "review"], verbose=True)
    response = overall_chain({"chat_history" : chat_history, "questions": questions})
    response_text = response["review"]
    response_array = response_text.split("\n\n")
    print(response_array)
    return response_array
