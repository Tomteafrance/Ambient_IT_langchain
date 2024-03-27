from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from pprint import pprint

tools = [TavilySearchResults(max_results=1)]

# Choose the LLM that will drive the agent
# Only certain models support this
chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

# Adapted from https://smith.langchain.com/hub/hwchase17/openai-tools-agent

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu es un assistant IA qui réponds aux questions de l'utilisateur en 20 mots ou moins. Tu réponds aux question dans la langue auquel on te l'a posé",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

agent = create_openai_tools_agent(chat, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
demo_ephemeral_chat_history_for_chain = ChatMessageHistory()

conversational_agent_executor = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="input",
    output_messages_key="output",
    history_messages_key="chat_history",
)

chunks=[]#optional

print("Bonjour je suis votre assistant IA. Comment puis-je vous aider?")
for i in range(5):
    query = input()
    if query=="clear":
        break
    for chunk in conversational_agent_executor.stream(
        {"input": query,},{"configurable": {"session_id": "unused"}}
    ):
        chunks.append(chunk) #optional
        print("------")
        pprint(chunk, depth=1)