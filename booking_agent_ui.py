import chainlit as cl
import os, getpass
env_path = r'C:\Users\VR679RX\OneDrive - EY\Documents\Projects\Projects - 2024\LangGraph_Academy\gpt4o.env'
from dotenv import load_dotenv
import json
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
load_dotenv(env_path)
from datetime import datetime
from langchain_core.pydantic_v1 import constr, BaseModel, Field, validator
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langchain_core.prompts.chat import ChatPromptTemplate,MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import TavilySearchResults
import os
from langgraph.prebuilt import create_react_agent
from typing import  Literal
from langchain_core.tools import tool
from langchain.agents.openai_tools.base import create_openai_tools_agent
from langchain.agents.agent import AgentExecutor
import shutil
import sqlite3
import functools
import pandas as pd
import requests

llm = AzureChatOpenAI(temperature=0,
                           api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                           azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                           openai_api_version=os.getenv('AZURE_OPENAI_VERSION'),
                           azure_deployment=os.getenv('AZURE_GPT4O_MODEL')
                           )

embeddings = AzureOpenAIEmbeddings(
                            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                            azure_deployment=os.getenv('AZURE_OPENAI_EMBEDDINGS_MODEL'),
                            openai_api_version=os.getenv('AZURE_OPENAI_VERSION'),
                            )

from langchain_core.pydantic_v1 import constr, BaseModel, Field, validator
import re


class DateTimeModel(BaseModel):
    """
    The way the date should be structured and formatted
    """
    date: str = Field(..., description="Propertly formatted date", pattern=r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$')

    @validator("date")
    def check_format_date(cls, v):
        if not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$', v):
            raise ValueError("The date should be in format 'YYYY-MM-DD HH:MM'")
        return v
    
class DateModel(BaseModel):
    """
    The way the date should be structured and formatted
    """
    date: str = Field(..., description="Propertly formatted date", pattern=r'^\d{2}-\d{2}-\d{4}$')

    @validator("date")
    def check_format_date(cls, v):
        if not re.match(r'^\d{2}-\d{2}-\d{4}$', v):
            raise ValueError("The date must be in the format 'YYYY-MM-DD'")
        return v

    
class IdentificationNumberModel(BaseModel):
    """
    The way the ID should be structured and formatted
    """
    id: int = Field(..., description="identification number without dots", pattern=r'^\d{7,8}$')

    @validator("id")
    def check_format_id(cls, v):
        if not re.match(r'^\d{7,8}$',str(v)):
            raise ValueError("The ID number should be a number of 7 or 8 numbers")
        return v
    

@tool
def check_availability_by_doctor(desired_date:DateModel, doctor_name:Literal['kevin anderson','robert martinez','susan davis','daniel miller','sarah wilson','michael green','lisa brown','jane smith','emily johnson','john doe']):
    """
    Checking the database if we have availability for the specific doctor.
    The parameters should be mentioned by the user in the query
    """
    #Dummy data
    df = pd.read_csv(r"C:\Users\VR679RX\OneDrive - EY\Documents\Projects\Projects - 2024\LangGraph_Academy\repos\appointment_agent\availability.csv")
    df['date_slot_time'] = df['date_slot'].apply(lambda input: input.split(' ')[-1])
    rows = list(df[(df['date_slot'].apply(lambda input: input.split(' ')[0]) == desired_date.date)&(df['doctor_name'] == doctor_name)&(df['is_available'] == True)]['date_slot_time'])

    if len(rows) == 0:
        output = "No availability in the entire day"
    else:
        output = f'This availability for {desired_date.date}\n'
        output += "Available slots: " + ', '.join(rows)

    return output

@tool
def check_availability_by_specialization(desired_date:DateModel, specialization:Literal["general_dentist", "cosmetic_dentist", "prosthodontist", "pediatric_dentist","emergency_dentist","oral_surgeon","orthodontist"]):
    """
    Checking the database if we have availability for the specific specialization.
    The parameters should be mentioned by the user in the query
    """
    #Dummy data
    df = pd.read_csv(r"C:\Users\VR679RX\OneDrive - EY\Documents\Projects\Projects - 2024\LangGraph_Academy\repos\appointment_agent\availability.csv")
    df['date_slot_time'] = df['date_slot'].apply(lambda input: input.split(' ')[-1])
    rows = df[(df['date_slot'].apply(lambda input: input.split(' ')[0]) == desired_date.date) & (df['specialization'] == specialization) & (df['is_available'] == True)].groupby(['specialization', 'doctor_name'])['date_slot_time'].apply(list).reset_index(name='available_slots')

    if len(rows) == 0:
        output = "No availability in the entire day"
    else:
        def convert_to_am_pm(time_str):
            # Split the time string into hours and minutes
            time_str = str(time_str)
            hours, minutes = map(int, time_str.split("."))
            
            # Determine AM or PM
            period = "AM" if hours < 12 else "PM"
            
            # Convert hours to 12-hour format
            hours = hours % 12 or 12
            
            # Format the output
            return f"{hours}:{minutes:02d} {period}"
        output = f'This availability for {desired_date.date}\n'
        for row in rows.values:
            output += row[1] + ". Available slots: \n" + ', \n'.join([convert_to_am_pm(value)for value in row[2]])+'\n'

    return output

def convert_datetime_format(dt_str):
    # Parse the input datetime string
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
    
    # Format the output as 'DD-MM-YYYY H.M' (removing leading zero from hour only)
    return dt.strftime("%d-%m-%Y %#H.%M")

@tool
def reschedule_appointment(old_date:DateTimeModel, new_date:DateTimeModel, id_number:IdentificationNumberModel, doctor_name:Literal['kevin anderson','robert martinez','susan davis','daniel miller','sarah wilson','michael green','lisa brown','jane smith','emily johnson','john doe']):
    """
    Rescheduling an appointment.
    The parameters MUST be mentioned by the user in the query.
    """
    #Dummy data
    df = pd.read_csv(r"C:\Users\VR679RX\OneDrive - EY\Documents\Projects\Projects - 2024\LangGraph_Academy\repos\appointment_agent\availability.csv")
    available_for_desired_date = df[(df['date_slot'] == convert_datetime_format(new_date.date))&(df['is_available'] == True)&(df['doctor_name'] == doctor_name)]
    if len(available_for_desired_date) == 0:
        return "Not available slots in the desired period"
    else:
        cancel_appointment.invoke({'date':old_date, 'id_number':id_number, 'doctor_name':doctor_name})
        set_appointment.invoke({'desired_date':new_date, 'id_number': id_number, 'doctor_name': doctor_name})
        return "Succesfully rescheduled for the desired time"

@tool
def cancel_appointment(date:DateTimeModel, id_number:IdentificationNumberModel, doctor_name:Literal['kevin anderson','robert martinez','susan davis','daniel miller','sarah wilson','michael green','lisa brown','jane smith','emily johnson','john doe']):
    """
    Canceling an appointment.
    The parameters MUST be mentioned by the user in the query.
    """
    df = pd.read_csv(r"C:\Users\VR679RX\OneDrive - EY\Documents\Projects\Projects - 2024\LangGraph_Academy\repos\appointment_agent\availability.csv")
    case_to_remove = df[(df['date_slot'] == convert_datetime_format(date.date))&(df['patient_to_attend'] == id_number.id)&(df['doctor_name'] == doctor_name)]
    if len(case_to_remove) == 0:
        return "You don´t have any appointment with that specifications"
    else:
        df.loc[(df['date_slot'] == convert_datetime_format(date.date)) & (df['patient_to_attend'] == id_number.id) & (df['doctor_name'] == doctor_name), ['is_available', 'patient_to_attend']] = [True, None]
        df.to_csv(r"C:\Users\VR679RX\OneDrive - EY\Documents\Projects\Projects - 2024\LangGraph_Academy\repos\appointment_agent\availability.csv", index = False)

        return "Succesfully cancelled"
    

@tool
def set_appointment(desired_date:DateTimeModel, id_number:IdentificationNumberModel, doctor_name:Literal['kevin anderson','robert martinez','susan davis','daniel miller','sarah wilson','michael green','lisa brown','jane smith','emily johnson','john doe']):
    """
    Set appointment or slot with the doctor.
    The parameters MUST be mentioned by the user in the query.
    """
    df = pd.read_csv(r"C:\Users\VR679RX\OneDrive - EY\Documents\Projects\Projects - 2024\LangGraph_Academy\repos\appointment_agent\availability.csv")
    from datetime import datetime



    
    case = df[(df['date_slot'] == convert_datetime_format(desired_date.date))&(df['doctor_name'] == doctor_name)&(df['is_available'] == True)]
    if len(case) == 0:
        return "No available appointments for that particular case"
    else:
        df.loc[(df['date_slot'] == convert_datetime_format(desired_date.date))&(df['doctor_name'] == doctor_name) & (df['is_available'] == True), ['is_available','patient_to_attend']] = [False, id_number.id]
        df.to_csv(r"C:\Users\VR679RX\OneDrive - EY\Documents\Projects\Projects - 2024\LangGraph_Academy\repos\appointment_agent\availability.csv", index = False)

        return "Succesfully done"

from typing import Literal
from typing_extensions import TypedDict, Annotated
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

def create_agent(llm:AzureChatOpenAI,tools:list,system_prompt:str):
    system_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt
                ),
                ("placeholder", "{messages}"),
            ]
        )
    agent = create_react_agent(model=llm,tools=tools,prompt=system_prompt)
    return agent

class AgentState(MessagesState):
    id_number:IdentificationNumberModel
    query:str
    next:str
    cur_reasoning:str

information_agent = create_agent(
                        llm=llm,
                        tools=[check_availability_by_doctor,check_availability_by_specialization],
                        system_prompt = "You are specialized agent to provide information related to availbility of doctors or any FAQs related to hospital based on the query. You have access to the tool.\n Make sure to ask user politely if you need any further information to execute the tool.\n For your information, Always consider current year is 2024."
                )

booking_agent = create_agent(
                        llm=llm,
                        tools=[set_appointment,cancel_appointment,reschedule_appointment],
                        system_prompt = "You are specialized agent to set, cancel or reschedule appointment based on the query. You have access to the tool.\n Make sure to ask user politely if you need any further information to execute the tool.\n For your information, Always consider current year is 2024."
                )

def information_node(state: AgentState):
    result = information_agent.invoke(state)
    return Command(
        update={
            "messages": state["messages"] + [
                AIMessage(content=result["messages"][-1].content, name="information_node")
            ]
        },
        goto="supervisor",
    )

def booking_node(state: AgentState):
    result = booking_agent.invoke(state)
    return Command(
        update={
            "messages": state["messages"] + [
                AIMessage(content=result["messages"][-1].content, name="booking_node")
            ]
        },
        goto="supervisor",
    )

members_dict = {'information_node':'specialized agent to provide information related to availbility of doctors or any FAQs related to hospital.','booking_node':'specialized agent to only to book, cancel or reschedule appointment'}
options = list(members_dict.keys()) + ["FINISH"]
worker_info = '\n\n'.join([f'WORKER: {member} \nDESCRIPTION: {description}' for member, description in members_dict.items()]) + '\n\nWORKER: FINISH \nDESCRIPTION: If User Query is answered and route to Finished'

system_prompt = (
    "You are a supervisor tasked with managing a conversation between following workers. "
    "### SPECIALIZED ASSISTANT:\n"
    f"{worker_info}\n\n"
    "Your primary role is to help the user make an appointment with the doctor and provide updates on FAQs and doctor's availability. "
    "If a customer requests to know the availability of a doctor or to book, reschedule, or cancel an appointment, "
    "delegate the task to the appropriate specialized workers. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
    "UTILIZE last conversation to assess if the conversation should end you answered the query, then route to FINISH "
     )

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH. and provide reasoning for the routing"""

    next: Annotated[Literal[*options], ..., "worker to route to next, route to FINISH"]
    reasoning: Annotated[str, ..., "Support proper reasoning for routing to the worker"]

def supervisor_node(state: AgentState) -> Command[Literal[*list(members_dict.keys()), "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
        #{"role": "user", "content": f"user's identification number is {state['id_number']}"},
    ] + [state["messages"][-1]]
    query = ''
    if len(state['messages'])==1:
        query = state['messages'][0].content
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END
    if query:
        return Command(goto=goto, update={"next": goto,'query':query,'cur_reasoning':response["reasoning"],
                                    "messages":[HumanMessage(content=f"user's identification number is {state['id_number']}")]
                        })
    return Command(goto=goto, update={"next": goto,'cur_reasoning':response["reasoning"]})


@cl.on_chat_start
async def on_chat_start():
    # start graph

    # Graph
    workflow = StateGraph(AgentState)
    workflow.add_edge(START, "supervisor")
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("information_node", information_node)
    workflow.add_node("booking_node", booking_node)
    memory = MemorySaver()
    # graph_v1 = workflow.compile(checkpointer=memory,)

    # initialize state
    #state = MessagesState(messages=[])

    # save graph and state to the user session
    cl.user_session.set("graph", workflow.compile(checkpointer=memory))
    #cl.user_session.set("state", state)


@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve the graph and state from the user session
    graph: Runnable = cl.user_session.get("graph")
    state = cl.user_session.get("state")

    # Append the new message to the state
    #state["messages"] += [HumanMessage(content=message.content)]
    state = {'messages': [HumanMessage(content=message.content)],'id_number':10232303}
    config = {"configurable": {"thread_id": "1", "recursion_limit": 10}}  
    # Stream the response to the UI
    # ui_message = cl.Message(content="")
    # await ui_message.send()
        
    events = []
    # results = graph.invoke(state,config=config)
    #await cl.Message(content=results['messages'][-1].content).send()
    event_dict = {}
    event_result = ''
    async for event in graph.astream_events(input=state,config=config, version="v1"):
        if event['event']=='on_chain_end' and event['name'] in ['supervisor'] + options:
            event_dict[event['run_id']] = cl.Step(name=f'Agent - {event['name']}', type="tool")
            event_result = event['data']['input']['messages'][-1].content
            await event_dict[event['run_id']].send()
    await cl.Message(content=event_result).send()

    #await ui_message.update()