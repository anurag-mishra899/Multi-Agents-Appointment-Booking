from langchain_core.prompts.chat import ChatPromptTemplate

def get_runnable(llm,tools,agent_prompt):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
            "system",
            agent_prompt
            ),
            ("placeholder", "{messages}"),
        ]
    )

    agent_runnable = prompt_template | llm.bind_tools(tools)
    return agent_runnable


