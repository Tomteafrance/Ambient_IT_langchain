from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_exa import ExaSearchRetriever
from langchain_openai import ChatOpenAI

#Init ExaSearchRetriever
retriever = ExaSearchRetriever(k=3, highlights=True)

#Prompt Template 
document_prompt = PromptTemplate.from_template(
    """
<source>
    <url>{url}</url>
    <highlights>{highlights}</highlights>
</source>
"""
)

document_chain = (
    RunnableLambda(
        lambda document: {
            "highlights": document.metadata["highlights"],
            "url": document.metadata["url"],
        }
    )
    | document_prompt
)

retrieval_chain = (
    retriever | document_chain.map() | (lambda docs: "\n".join([i.text for i in docs]))
)


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Vous êtes un assistant de recherche expert. Vous utilisez du contexte en format xml pour répondre aux questions des utilisateurs.",
        ),
        (
            "human",
            """
Veuillez répondre à la question suivante en vous basant sur le contexte fourni. Veuillez citer vos sources à la fin de votre réponse.:
     
Query: {query}
---
<context>
{context}
</context>
""",
        ),
    ]
)

llm = ChatOpenAI()

chain = (
    RunnableParallel(
        {
            "query": RunnablePassthrough(),
            "context": retrieval_chain,
        }
    )
    | generation_prompt
    | llm
).with_types(input_type=str)


