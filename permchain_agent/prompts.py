from langchain.prompts import ChatPromptTemplate
from agent.prompts import auto_agent_instructions
from langchain.schema.messages import SystemMessage
summary_message = (
    '"""{chunk}""" Using the above text, answer the following'
        ' question: "{question}" -- if the question cannot be answered using the text,'
        " simply summarize the text in depth. "
        "Include all factual information, numbers, stats etc if available."
)
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("user", summary_message)
])

search_message = (
'Write 4 google search queries to search online that form an objective opinion from the following: "{question}"'\
           f'You must respond with a list of strings in the following format: ["query 1", "query 2", "query 3", "query 4"]'

)

SEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{agent_prompt}"),
    ("user", search_message)
])

CHOOSE_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=auto_agent_instructions()),
("user", "task: {task}")
])

report_message = (
    '"""{research_summary}""" Using the above information, answer the following'\
           ' question or topic: "{question}" in a detailed report --'\
           " The report should focus on the answer to the question, should be well structured, informative," \
           " in depth, with facts and numbers if available, a minimum of 1,200 words and with markdown syntax and apa format. "\
            "You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions." \
           "Write all used source urls at the end of the report in apa format"
)


REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{agent_prompt}"),
    ("user", report_message)
])