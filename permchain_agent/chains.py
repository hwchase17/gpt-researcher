from operator import itemgetter

from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.runnables.openai_functions import OpenAIFunctionsRouter
import json
from langchain.schema.runnable import RunnableMap
from permchain.connection_inmemory import InMemoryPubSubConnection
from permchain.pubsub import PubSub
from permchain.topic import Topic
from permchain_agent.prompts import SUMMARY_PROMPT, SEARCH_PROMPT, CHOOSE_AGENT_PROMPT, REPORT_PROMPT
from processing.text import split_text
from actions.web_scrape import scrape_text_with_selenium
from actions.web_search import web_search


summary_chain = SUMMARY_PROMPT | ChatOpenAI() | StrOutputParser()
chunk_and_combine = (lambda x: [{
    "question": x["question"],
    "chunk": chunk
} for chunk in split_text(x["text"])]) | summary_chain.map() | (lambda x: "\n".join(x))

recursive_summary_chain = {
    "question": lambda x: x["question"],
    "chunk": chunk_and_combine
} | summary_chain

scrape_and_summarize = {
    "question": lambda x: x["question"],
    "text": lambda x: scrape_text_with_selenium(x['url'])[1]
} | recursive_summary_chain

multi_search = (lambda x: [
    {"url": url.get("href"), "question": x["question"]}
    for url in json.loads(web_search(x["question"]))
]) | scrape_and_summarize.map() | (lambda x: "\n".join(x))

search_query = SEARCH_PROMPT | ChatOpenAI() |  StrOutputParser() | json.loads
choose_agent = CHOOSE_AGENT_PROMPT | ChatOpenAI() | StrOutputParser() | json.loads
report_chain = REPORT_PROMPT | ChatOpenAI() | StrOutputParser()

get_search_queries = {
    "question": lambda x: x["question"],
    "agent_prompt": {"task": lambda x: x["question"]} | choose_agent | (lambda x: x["agent_role_prompt"])
} | search_query

full_chain = RunnableMap({
    "question": lambda x: x["question"],
    "agent_prompt": {"task": lambda x: x["question"]} | choose_agent | (lambda x: x["agent_role_prompt"])
}) | {
    "question": lambda x: x["question"],
    "agent_prompt": lambda x: x["agent_prompt"],
    "research_summary": get_search_queries | (lambda x: [{"question": q} for q in x]) | multi_search.map() | (lambda x: "\n\n".join(x))
} | report_chain
if __name__ == "__main__":
    # print(multi_search.invoke({
    #     #"url": "https://blog.langchain.dev/chatopensource-x-langchain-the-future-is-fine-tuning-2/",
    #     "question": "how can langsmith help with finetuning"
    # }))

    print(full_chain.invoke({
        "question": "how can langsmith help with finetuning",
    }))
