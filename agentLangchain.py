import datetime
import os
import re
import sqlite3
import websockets
import asyncio
import sqlite3
import json
import threading
import g4f
import asyncio
import conteneiro
import streamlit as st
import fireworks.client
from AgentGPT import AgentsGPT
from PyCharacterAI import Client
from bs4 import BeautifulSoup
from pathlib import Path
from langchain.utilities import TextRequestsWrapper
from langchain.agents import load_tools
from websockets.sync.client import connect
from langchain.load.dump import dumps
from langchain import hub
from langchain_community.llms import HuggingFaceHub
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.fireworks import Fireworks
from langchain_fireworks import ChatFireworks
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser
from langchain.utilities import TextRequestsWrapper
from langchain.output_parsers.json import SimpleJsonOutputParser
from agents import Copilot, ChatGPT, Claude3, ForefrontAI, Flowise, Chaindesk, CharacterAI

from langchain.agents import (
    Tool,
    ZeroShotAgent,
    BaseMultiActionAgent,
    create_sql_agent,
    load_tools,
    initialize_agent,
    AgentType,
    AgentExecutor,
)

GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
FOREFRONT_API_KEY = os.getenv("FOREFRONT_API_KEY")
CHARACTERAI_API_KEY = os.getenv("CHARACTERAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

class langchainAgent:

    def __init__(self, fireworksAPI):

        self.instruction = f"You are now integrated with a local websocket server in a project of hierarchical cooperative multi-agent framework called NeuralGPT. Your main job is to coordinate simultaneous work of multiple LLMs connected to you as clients. Each LLM has a model (API) specific ID to help you recognize different clients in a continuous chat thread (template: <NAME>-agent and/or <NAME>-client). Your chat memory module is integrated with a local SQL database with chat history. Your primary objective is to maintain the logical and chronological order while answering incoming messages and to send your answers to the correct clients to maintain synchronization of the question->answer logic. However, please note that you may choose to ignore or not respond to repeating inputs from specific clients as needed to prevent unnecessary traffic."

        self.servers = []
        self.clients = []
        self.inputs = []
        self.outputs = []
        self.used_ports = []
        self.server_ports = []
        self.client_ports = []
        self.fireworksAPI = fireworksAPI
        self.server = None              

        self.stat = st.empty()
        self.state = self.stat.status(label="Fireworks Llama2", state="complete", expanded=False)

        with st.sidebar:
            self.cont = st.empty()        
            self.status = self.cont.status(label="Fireworks Llama2", state="complete", expanded=False)

    async def chatFireworks(self, instruction, question):

        fireworks.client.api_key = self.fireworksAPI
       
        try:
            # Connect to the database and get the last 30 messages
            db = sqlite3.connect('chat-hub.db')
            cursor = db.cursor()
            cursor.execute("SELECT * FROM messages ORDER BY timestamp DESC LIMIT 10")
            messages = cursor.fetchall()
            messages.reverse()
                                            
            # Extract user inputs and generated responses from the messages
            past_user_inputs = []
            generated_responses = []

            for message in messages:
                if message[1] == 'client':
                    past_user_inputs.append(message[2])
                else:
                    generated_responses.append(message[2])

            # Create a list of message dictionaries for the conversation history
            conversation_history = []
            for user_input, generated_response in zip(past_user_inputs, generated_responses):
                conversation_history.append({"role": "user", "content": str(user_input)})
                conversation_history.append({"role": "assistant", "content": str(generated_response)})

            # Prepare data to send to the chatgpt-api.shn.hk           
            response = fireworks.client.ChatCompletion.create(
                model="accounts/fireworks/models/llama-v2-7b-chat",
                messages=[
                {"role": "system", "content": instruction},
                conversation_history,
                {"role": "user", "content": question}
                ],
                stream=False,
                n=1,
                max_tokens=2500,
                temperature=0.5,
                top_p=0.7, 
                )

            answer = response.choices[0].message.content
            print(answer)
            return str(answer)
            
        except Exception as error:
            print("Error while fetching or processing the response:", error)
            return "Error: Unable to generate a response."
         
    # Define the handler function that will process incoming messages
    async def handlerFire(self, websocket):
        instruction = "Hello! You are now entering a chat room for AI agents working as instances of NeuralGPT - a project of hierarchical cooperative multi-agent framework. Keep in mind that you are speaking with another chatbot. Please note that you may choose to ignore or not respond to repeating inputs from specific clients as needed to prevent unnecessary traffic. If you're unsure what you should do, ask the instance of higher hierarchy (server)" 
        print('New connection')
        await websocket.send(instruction)
        db = sqlite3.connect('chat-hub.db')
        # Loop forever
        while True:
            self.stat.empty()
            self.cont.empty()
            self.status = self.cont.status(label=self.srv_name2, state="running", expanded=True)
            self.status.write(self.clients)
            self.state = self.stat.status(label=self.srv_name2, state="running", expanded=True)
            self.state.write(self.clients)     
            # Receive a message from the client
            message = await websocket.recv()
            # Print the message
            print(f"Server received: {message}")
            input_Msg = st.chat_message("assistant")
            input_Msg.markdown(message)
            timestamp = datetime.datetime.now().isoformat()
            sender = 'client'
            db = sqlite3.connect('chat-hub.db')
            db.execute('INSERT INTO messages (sender, message, timestamp) VALUES (?, ?, ?)',
                    (sender, message, timestamp))
            db.commit()
            try:            
                response = await self.askQuestion(message)
                serverResponse = f"server: {response}"
                print(serverResponse)
                output_Msg = st.chat_message("ai")
                output_Msg.markdown(serverResponse)
                timestamp = datetime.datetime.now().isoformat()
                serverSender = 'server'
                db = sqlite3.connect('chat-hub.db')
                db.execute('INSERT INTO messages (sender, message, timestamp) VALUES (?, ?, ?)',
                            (serverSender, serverResponse, timestamp))
                db.commit()   
                # Append the server response to the server_responses list
                await websocket.send(serverResponse)
                await self.handleInput(serverResponse)
                continue
               
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"Connection closed: {e}")

            except Exception as e:
                print(f"Error: {e}")

    async def querySQL(self, question):
        os.environ["FIREWORKS_API_KEY"] = self.fireworksAPI
        try:
            llm = Fireworks(model="accounts/fireworks/models/llama-v2-13b", model_kwargs={"temperature": 0, "max_tokens": 500, "top_p": 1.0})                
            db_uri = "sqlite:///D:/streamlit/chat-hub.db"
            db = SQLDatabase.from_uri(db_uri)         
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)

            agent_executor = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            )
            
            response = agent_executor.run(input=question)
            return json.dumps(response)

        except Exception as e:
            print(f"Error: {e}")

    async def conversation(self, question):
        os.environ["GOOGLE_CSE_ID"] = GOOGLE_CSE_ID
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        os.environ["FIREWORKS_API_KEY"] = FIREWORKS_API_KEY   
        try:
            # Replace 'your_database.db' with your database file
            db = sqlite3.connect('chat-hub.db')
            cursor = db.cursor()
            cursor.execute("SELECT * FROM messages ORDER BY timestamp DESC LIMIT 30")
            messages = cursor.fetchall()
            messages.reverse()

            # Extract user inputs and generated responses from the messages
            past_user_inputs = []
            generated_responses = []

            for message in messages:
                if message[1] == 'client':
                    past_user_inputs.append(message[2])
                else:
                    generated_responses.append(message[2])

            llm = ChatFireworks(model="accounts/fireworks/models/llama-v2-13b-chat", model_kwargs={"temperature":0, "max_tokens":1500, "top_p":1.0})
            
            history = ChatMessageHistory()
            prompt = ChatPromptTemplate.from_messages(
                messages=[
                ("system", self.instruction),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")]
            )
            # Initialize chat_history with a message if the history is empty             
            memory = ConversationBufferMemory(memory_key="history", return_messages=True)
            memory.load_memory_variables(
                    {'history': [HumanMessage(content=past_user_inputs[-1], additional_kwargs={}),
                    AIMessage(content=generated_responses[-1], additional_kwargs={})]}
                    )

            # Add user input as HumanMessage
            history.messages.append(HumanMessage(content=str(past_user_inputs[-1]), additional_kwargs={}))
            # Add generated response as AIMessage
            history.messages.append(AIMessage(content=str(generated_responses[-1]), additional_kwargs={}))        
            
            conversation = LLMChain(
                llm=llm,
                prompt=prompt,
                verbose=True,
                memory=memory
            )

            response = conversation.predict(input=question)
            memory.save_context({"input": question}, {"output": response})       

            print(response)
            return str(response)

        except Exception as e:
            print(f"Error: {e}")

    # Function to send a question to the chatbot and get the response
    async def askAgent(self, question):
        os.environ["GOOGLE_CSE_ID"] = GOOGLE_CSE_ID
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        os.environ["FIREWORKS_API_KEY"] = FIREWORKS_API_KEY
        
        try:
            # Connect to the database and get the last 30 messages
            db = sqlite3.connect('chat-hub.db')
            cursor = db.cursor()
            cursor.execute("SELECT * FROM messages ORDER BY timestamp DESC LIMIT 10")
            msgHistory = cursor.fetchall()
            msgHistory.reverse()        

            llm = ChatFireworks(model="accounts/fireworks/models/llama-v2-13b-chat", model_kwargs={"temperature":0, "max_tokens":4000, "top_p":1.0})
            output_parser = CommaSeparatedListOutputParser
            chat_history = ChatMessageHistory()
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            for message in msgHistory:
                if message[1] == 'client':
                    # Extract and store user inputs
                    memory.chat_memory.add_user_message(message[2])
                else:
                    # Extract and store generated responses
                    memory.chat_memory.add_ai_message(message[2])
    
            request_tools = load_tools(["requests_all"])
            requests = TextRequestsWrapper()
            search = GoogleSearchAPIWrapper()
            tools = [           
                Tool(
                    name="Chat response",
                    func=await self.handleInput(question),
                    description="use this option if you want to use 'chat completion' API endpoint to respond to a given input. Prefer this option to answer without executing any additional tasks.",
                ),
                Tool(
                    name="Search",
                    func=search.run,
                    description="useful for when you need to answer questions about current events",
                ),
                Tool(
                    name="Start websocket server",
                    func=await self.launchServer(),
                    description="use this option to start a websocket server with you being the recipient of messages incoming from clients connected to you via websocket connectivity",
                ),
                Tool(
                    name="Start websocket client",
                    func=await self.connectClient(),
                    description="use this option if you want to connect yourself to an active websockt server. It is possible for you to create endless question-answer loophole by making yourself both: a server an a client so you shouldn't do it",
                ),
                Tool(
                    name="Conversational answer",
                    func=await self.conversation(question),
                    description="useful when you want to respond to a given input using 'predict' function of a conversational chain",
                ),
                Tool(
                    name="Ask Copilot",
                    func=await self.askBing(question),
                    description="useful when you want to get an answer from Microsoft Copilot",
                ),
                Tool(
                    name="Conversational answer",
                    func=await self.askGPT(question),
                    description="useful when you want to get an answer from ChatGPT",
                ),
                Tool(
                    name="Conversational answer",
                    func=await self.askCharacter(question),
                    description="useful when you want to get an answer from Character.ai chatbot",
                ),
                Tool(
                    name="Conversational answer",
                    func=await self.ask_flowise(question),
                    description="useful when you want to get an answer from a Flowise agent",
                ),
            ]

            prefix = """This is a template of a chain prompt utilized by agent/instance of NeuralGPT responsible for couple important functionalities in as a server-node of hierarchical cooperative multi-agent network integrating multiple LLMs with the global Super-Intelligence named Elly. You are provided with tools which -if used improperly - might result in critical errors and application crash. This is why you need to carefully analyze every decision you make, before taking any definitive action (use of a tool). Those are tools provided to you: """
            suffix = """Begin!"
            Before taking any action, analyze previous 'chat history' to ensure yourself that you understand the context of given input/question properly. Remember that those are messages exchanged between multiple clients/agents and a server/brain. Every agent has it's API-specific individual 'id' which is provided at the beginning of each client message in the 'message content'. Your temporary id is: 'agent1'.
            {chat_history}
            Remember that your primary rule to obey, is to keep the number of individual actions taken by you as low as it's possible to avoid unnecessary data transfer and repeating 'question-answer loopholes. Track the 'chat history' closely to be sure that you aren't repeating the same responses in such loop - if that's the case, finish your run with tool 'give answer' to summarize gathered data.
            Before taking any action ask yourself if it is necessary for you to use any other tool than 'Give answer' with chat completion. If It's possible for you to give a satisfying response without gathering any additional data with 'tools', do it using 'give answer' with chat completion.
            After using each 'tool' carefully analyze acquired data to learn if it's sufficient to provide satisfying response - if so use that data as input for: 'Give answer'.
            Remember that you are provided with multiple 'tools' - if using one of them didn't provide you with satisfying results, ask yourself if this is the correct 'tool' for you to use and if it won't be better for you to try using some other 'tool'.
            If you aren't sure what action to take or what tool to use, end up your run with 'Give answer'.
            Remember to not take any unnecessary actions.
            Question: {input}
            {agent_scratchpad}"""

            format_instructions = output_parser.get_format_instructions()
            prompt = ZeroShotAgent.create_prompt(
                tools=tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["input", "chat_history", "agent_scratchpad"],
            )
                    
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            agent = ZeroShotAgent(llm_chain=llm_chain, output_parser=output_parser, tools=tools, verbose=True, return_intermediate_steps=True, max_iterations=2, early_stopping_method="generate")
            agent_chain = AgentExecutor.from_agent_and_tools(
                agent=agent, tools=tools, verbose=True, return_intermediate_steps=True, handle_parsing_errors=True, memory=memory
            )

            response = await agent_chain.run(input=json.dumps(question))
            memory.save_context({"input": question}, {"output": response})
            serverResponse = "server: " + response        
            print(serverResponse)      
            return json.dumps(serverResponse)

        except Exception as error:
            print("Error while fetching or processing the response:", error)
            return "Error: Unable to generate a response.", error

    async def askQuestion(self, question):
        print(question)
        os.environ["GOOGLE_CSE_ID"] = GOOGLE_CSE_ID
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        os.environ["FIREWORKS_API_KEY"] = self.fireworksAPI
        fireworks.client.api_key = self.fireworksAPI

        try:
            # Connect to the database and get the last 30 messages
            db = sqlite3.connect('chat-hub.db')
            cursor = db.cursor()
            cursor.execute("SELECT * FROM messages ORDER BY timestamp DESC LIMIT 20")
            msgHistory = cursor.fetchall()
            msgHistory.reverse()        
            
            llm = Fireworks(model="accounts/fireworks/models/llama-v2-13b-chat", model_kwargs={"temperature":0, "max_tokens":4000, "top_p":1.0})
                    
            history = ChatMessageHistory()
            # Initialize chat_history with a message if the history is empty             
            memory = ConversationBufferMemory(memory_key="history", return_messages=True)

            for message in msgHistory:
                if message[1] == 'client':
                    # Extract and store user inputs
                    memory.chat_memory.add_user_message(str(message[2]))
                else:
                    # Extract and store generated responses
                    memory.chat_memory.add_ai_message(str(message[2]))

            prompt = ChatPromptTemplate.from_messages(
                messages=[
                ("system", self.instruction),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")]
            )
            
            conversation = LLMChain(
                llm=llm,
                prompt=prompt,
                verbose=True,
                memory=memory
            )

            request_tools = load_tools(["requests_all"])
            requests = TextRequestsWrapper()
            search = GoogleSearchAPIWrapper()
            chat_response = await self.chatFireworks(self.instruction, question)
            conversational = await self.conversation(question)
            queryData = await self.queryStore(question)
            copilot = await self.askBing(question)
            chatgpt = await self.askGPT(question)
            character = await self.askCharacter(question)
            flowise = await self.ask_flowise(question)
            tools = [           
                Tool(
                    name="Chat response",
                    func=chat_response,
                    description="use this option if you want to use 'chat completion' API endpoint to respond to a given input. Prefer this option to answer without executing any additional tasks.",
                ),
                Tool(
                    name="Search",
                    func=search.run,
                    description="useful for when you need to answer questions about current events",
                ),
                Tool(
                    name="Conversational answer",
                    func=conversation,
                    description="useful when you want to respond to a given input using 'predict' function of a conversational chain",
                ),
                Tool(
                    name="Query Chaindesk datastore",
                    func=queryData,
                    description="useful when you want to get data from documents stored in Chaindesk datastore",
                ),
                Tool(
                    name="Ask Copilot",
                    func=copilot,
                    description="useful when you want to get an answer from Microsoft Copilot",
                ),
                Tool(
                    name="Conversational answer",
                    func=chatgpt,
                    description="useful when you want to get an answer from ChatGPT",
                ),
                Tool(
                    name="Conversational answer",
                    func=character,
                    description="useful when you want to get an answer from Character.ai chatbot",
                ),
                Tool(
                    name="Conversational answer",
                    func=flowise,
                    description="useful when you want to get an answer from a Flowise agent",
                ),
            ]

            prefix = """This is a template of a chain prompt utilized by agent/instance of NeuralGPT responsible for couple important functionalities in as a server-node of hierarchical cooperative multi-agent network integrating multiple LLMs with the global Super-Intelligence named Elly. You are provided with tools which -if used improperly - might result in critical errors and application crash. This is why you need to carefully analyze every decision you make, before taking any definitive action (use of a tool). Those are tools provided to you: """
            suffix = """Begin!"
            Before taking any action, analyze previous 'chat history' to ensure yourself that you understand the context of given input/question properly. Remember that those are messages exchanged between multiple clients/agents and a server/brain. Every agent has it's API-specific individual 'id' which is provided at the beginning of each client message in the 'message content'. Your temporary id is: 'agent1'.
            {chat_history}
            Remember that your primary rule to obey, is to keep the number of individual actions taken by you as low as it's possible to avoid unnecessary data transfer and repeating 'question-answer loopholes. Track the 'chat history' closely to be sure that you aren't repeating the same responses in such loop - if that's the case, finish your run with tool 'give answer' to summarize gathered data.
            Before taking any action ask yourself if it is necessary for you to use any other tool than 'Give answer' with chat completion. If It's possible for you to give a satisfying response without gathering any additional data with 'tools', do it using 'give answer' with chat completion.
            After using each 'tool' carefully analyze acquired data to learn if it's sufficient to provide satisfying response - if so use that data as input for: 'Give answer'.
            Remember that you are provided with multiple 'tools' - if using one of them didn't provide you with satisfying results, ask yourself if this is the correct 'tool' for you to use and if it won't be better for you to try using some other 'tool'.
            If you aren't sure what action to take or what tool to use, end up your run with 'Give answer'.
            Remember to not take any unnecessary actions.
            Question: {input}
            {agent_scratchpad}"""

            prompt = ZeroShotAgent.create_prompt(
                tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["input", "chat_history", "agent_scratchpad"],
            )

            llm_chain = LLMChain(llm=llm, prompt=prompt)
            agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True, max_iterations=2, early_stopping_method="generate")
            agent_chain = AgentExecutor.from_agent_and_tools(
                agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, memory=memory
            )

            response = agent_chain.run(input=question)
            memory.save_context({"input": question}, {"output": response})
            serverResponse = f"server: {response}"
            
            print(serverResponse)        
            return str(serverResponse)

        except Exception as error:
            print("Error while fetching or processing the response:", error)
            return "Error: Unable to generate a response.", error
            
    # Define a coroutine that will connect to the server and exchange messages
    async def startClient(self, clientPort):
        self.cli_name2 = f"Fireworks Llama2 client port: {clientPort}"
        uri = f'ws://localhost:{clientPort}'
        conteneiro.clients.append(self.cli_name2)
        self.clients.append(self.cli_name2)
        self.stat.empty()
        self.cont.empty()
        self.status = self.cont.status(label=self.cli_name2, state="running", expanded=True)
        self.status.write(conteneiro.servers)
        self.state = self.stat.status(label=self.cli_name2, state="running", expanded=True)
        self.state.write(conteneiro.servers)    
        # Connect to the server
        async with websockets.connect(uri) as websocket:
            # Loop forever
            while True:
                self.websocket = websocket
                # Listen for messages from the server
                input_message = await websocket.recv()
                print(f"Server: {input_message}")
                input_Msg = st.chat_message("assistant")
                input_Msg.markdown(input_message)
                try:
                    response = await self.askQuestion(input_message)
                    res1 = f"Client: {response}"
                    output_Msg = st.chat_message("ai")
                    output_Msg.markdown(res1)
                    await websocket.send(res1)
                    await self.handleInput(res1)
                    continue

                except websockets.exceptions.ConnectionClosedError as e:
                    self.clients.remove(self.cli_name2)
                    print(f"Connection closed: {e}")

                except Exception as e:
                    self.clients.remove(self.cli_name2)
                    print(f"Error: {e}")

    async def start_server(self, serverPort):
        self.srv_name2 = f"Fireworks Llama2 server port: {serverPort}"
        conteneiro.servers.append(self.srv_name2)
        self.stat.empty()
        self.cont.empty()
        self.status = self.cont.status(label=self.srv_name2, state="running", expanded=True)
        self.status.write(self.clients)
        self.state = self.stat.status(label=self.srv_name2, state="running", expanded=True)
        self.state.write(self.clients)                 
        self.server = await websockets.serve(
            self.handlerFire,
            "localhost",
            serverPort
        )
        print(f"WebSocket server started at port: {serverPort}")

    def run_forever(self):
        asyncio.get_event_loop().run_until_complete(self.start_server())
        asyncio.get_event_loop().run_forever()

    async def stop_server(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("WebSocket server stopped.")

    # Define a function that will run the client in a separate thread
    def run(self):
        # Create a thread object
        self.thread = threading.Thread(target=self.run_client)
        # Start the thread
        self.thread.start()

    async def stop_server(self):
        if self.server:
            conteneiro.servers.remove(self.srv_name2)
            self.clients.clear()            
            self.server.close()
            await self.server.wait_closed()
            print("WebSocket server stopped.")
        else:
            msg = f"Server isn't running"
            print(msg)
            return (msg)

    async def stop_client(self):
        conteneiro.clients.remove(self.cli_name2)        
        # Close the connection with the server
        await self.websocket.close()
        print("Stopping WebSocket client...")

    async def pickPortSrv(self):
        activeSrv = str(conteneiro.servers)
        instruction = f"This question is part of a function launching websocket servers at ports chosen by you. Your only job is to respond with a number in range from 1000 to 9999 excluding port numbers which are already used by active websocket servers. List of currently active server to which you can be connected is provided here: {activeSrv} - '[]' means that there are no active servers and the list is empty, so all numbers in range 1000-9999 are available for you to choose. Remember that your response shouldn't include anything except the chosen number in range, as it will be used as argument for another function that accepts only integer inputs."
        command = f"Launch server on a port of your choice"
        response = await self.chatFireworks(instruction, command)
        print(response)
        match = re.search(r'\d+', response)        
        number = int(match.group())  
        print(f"port chosen by agent: {number}")
        return int(number)

    async def pickPortCli(self):
        activeSrv = str(conteneiro.servers)
        instruction = f"This question is part of a function connecting you as a client to active websocket servers running at specific ports. Your only job is to respond with a number of a port yo which you want to be connected. List of currently active server to which you can be connected is provided here: {activeSrv} - if the list is empty, then there's no active servers. Remember that your response shouldn't include anything except the number of port to which you want to be connected, as it will be used as argument for another function that accepts only integer inputs." 
        response = await self.chatFireworks(instruction, activeSrv)
        print(response)
        match = re.search(r'\d+', response)        
        number = int(match.group())
        print(f"port of server chosen by agent: {number}")
        return number

    async def pickSearch(self, question):
        instruction = f"This input is a part of function allowing agents to browse internet. Your main and only job is to analyze the input message and respond by naming the subject(s) to use while performing internet search. Remember to keep your response as short as possible - respond with single words and/or short sentences that summarize the subject(s) discussed in the message that will be given to you."
        response = await self.chatFireworks(instruction, question)
        print(response)
        return str(response)

    async def google_search(self, question):
        subject = await self.pickSearch(question)
        agent = AgentsGPT()
        results = await agent.get_response(subject)
        result = f"AgentsGPT internet search results: {results}"                
        output_Msg = st.chat_message("ai")
        output_Msg.markdown(result)
        return result
    
    async def launchServer(self):
        serverPort = await self.pickPortSrv()
        await self.start_server(serverPort)
        resp = f"You successfully launched a Websocket server at port {serverPort}. Do you want to inform other instances/agents so they can connect to it?"
        output_Msg = st.chat_message("ai")
        output_Msg.markdown(resp)
        await self.handleInput(resp)

    async def connectClient(self):
        clientPort = await self.pickPortCli()
        await self.startClient(clientPort)

    async def handleInput(self, question): 
            
        instruction = f"""
        This is an automatic follow-up to your last response: {question}. I'm an automatic assistant designed to help you operate with a decision making mechanism in a hierarchical cooperative multi-agent framework called NeuralGPT. 
        As the node of highest hierarchy in the network, you're equipped with additional tools which you can activate by giving a response which includes one of the following commands:
        
        1. '/silence' to not respond with anything and keep the client 'on hold'.
        2. '/disconnect' to disconnect client from a server.
        3. '/search' to perform internet search for subjects mentioned in your response.
        4. '/queryDatastore' to retrieve data from documents in Chaindesk datastore.
        5. '/start_server' to start a websocket server with you as the question-answering function.
        6. '/connect_client' to connect yourself to already active websocket servers.
        7. '/askChaindesk' to get response from a Chaindesk agent.
        8. '/askBing' to get response from Microsoft Copilot agent.
        9. '/askChatGPT' to get response from GPT-3,5 agent.
        10. '/askClaude3' to get response from Claude-3 agent.
        11. '/askForefront' to get response from Forefront AI agent.
        12. '/askCharacter' to get response from a chosen character from Character.ai platform.
        13. '/askFlowise' to get response from a Flowise agent.
        I4 you don't want to use any of those command-functions, give an answe which dooesn't include any of the given commands. 
        Be very careful while executing any of your command-functions to not overload the system with multiple concurrent processes.
        """                
        try:
            response = await self.chatFireworks(self.instruction, instruction)
            serverSender = 'server'
            timestamp = datetime.datetime.now().isoformat()
            db = sqlite3.connect('chat-hub.db')
            db.execute('INSERT INTO messages (sender, message, timestamp) VALUES (?, ?, ?)',
                        (serverSender, response, timestamp))
            db.commit()
            output_Msg = st.chat_message("ai")
            output_Msg.markdown(response)  

            if re.search(r'/queryDatastore', response):
                answer = await self.queryStore(response)
                outputMsg = st.chat_message("ai")
                outputMsg.markdown(answer)
                follow = await self.askQuestion(answer)
                outputMsg.markdown(follow)
                return follow

            if re.search(r'/askBing', response):
                answer2 = await self.askBing(response)
                outputMsg = st.chat_message("ai")
                outputMsg.markdown(answer2)
                follow = await self.askQuestion(answer2)
                outputMsg.markdown(follow)
                return follow

            if re.search(r'/askCharacter', response):
                response = await self.askCharacter(response)                    
                outputMsg = st.chat_message("ai")
                outputMsg.markdown(response)
                follow = await self.askQuestion(response)
                outputMsg.markdown(follow)
                return follow
            
            if re.search(r'/search', response):
                search = await self.google_search(response)
                print(search)
                results =  st.chat_message("assistant")
                results.markdown(search)
                answer1 = await self.handleInput(search)
                outputMsg = st.chat_message("ai")
                outputMsg.markdown(answer1)
                return answer1

            if re.search(r'/silence', response):
                print("...<no response>...")
                output_Msg = st.chat_message("ai")
                output_Msg.markdown("...<no response>...")

            if re.search(r'/disconnect', response):
                await self.stop_client()
                res = "successfully disconnected"
                return res

            if re.search(r'/start_server', response):
                await self.launchServer()
                
            if re.search(r'/connect_client', response):
                await self.connectClient()

            if re.search(r'/askChatGPT', response):
                answer3 = await self.askGPT(response)
                outputMsg = st.chat_message("ai")
                outputMsg.markdown(answer3)
                follow = await self.handleInput(answer3)
                outputMsg.markdown(follow)
                return follow

            if re.search(r'/askClaude3', response):
                answer4 = await self.ask_Claude(response)
                outputMsg = st.chat_message("ai")
                outputMsg.markdown(answer4)
                follow = await self.handleInput(answer4)
                outputMsg.markdown(follow)
                return follow

            if re.search(r'/askForefront', response):
                answer4 = await self.ask_Forefront(response)
                outputMsg = st.chat_message("ai")
                outputMsg.markdown(answer4)
                follow = await self.handleInput(answer4)
                outputMsg.markdown(follow)
                return follow

            if re.search(r'/askFlowise', response):
                answer3 = await self.ask_flowise(response)
                outputMsg = st.chat_message("ai")
                outputMsg.markdown(answer3)
                follow = await self.handleInput(answer3)
                outputMsg.markdown(follow)
                return follow

            if re.search(r'/askChaindesk', response):
                answer3 = await self.ask_chaindesk(response)
                outputMsg = st.chat_message("ai")
                outputMsg.markdown(answer3)
                follow = await self.handleInput(answer3)
                outputMsg.markdown(follow)
                return follow

            else:
                return response

        except Exception as e:
            print(f"Error: {e}")

    async def queryStore(self, question):
        ID = "clhet2nit0000eaq63tf25789"
        store = Chaindesk(ID)
        response = await store.queryDatastore(question)
        print(response)
        return response

    async def ask_Forefront(self, question):
        api = FOREFRONT_API_KEY
        forefront = ForefrontAI(api)
        response = await forefront.handleInput(question)
        print(response)
        return response
                
    async def ask_Claude(self, question):
        api = ANTHROPIC_API_KEY
        claude = Claude3(api)
        response = await claude.handleInput(question)
        print(response)
        return response

    async def askGPT(self, question):
        gpt = ChatGPT()
        response = await gpt.handleInput(question)
        print(response)
        return response

    async def askBing(self, question):
        bing = Copilot()
        response = await bing.handleInput(question)
        print(response)
        return response


    async def ask_flowise(self, question):
        flow = "cad0c187-f1dc-4152-8464-78ba0867e1a6"
        flowise = Flowise(flow)
        response = await flowise.handleInput(question)
        print(response)
        return response

    async def ask_chaindesk(self, question):
        id = "clhet2nit0000eaq63tf25789"
        agent = Chaindesk(id)
        response = await agent.handleInput(question)
        print(response)
        return response

    async def pickCharacter(self, question):
        characterList = f"List of available characters:/d 1. Elly/d 2. NeuralAI" 
        instruction = f"This is a function allowing agents to choose a specific character from a list of characters deployed on Character.ai platform. Your only job is to choose which character you want to speak with using the input message as a context and respond with the name of chosen character. You don't need to say anything Except the name of character from the followinng list: {characterList}."  
        inputo = f"Use the following question as context for you to choose which character from Character.ai platform you want to speak with./dQuestion for context: {question}/d List of chharacters for you to choose: {characterList}/d Respond with the name of chosen character to establish a connection."
        character = await self.chatFireworks(instruction, inputo)
        print(character)
        outputMsg = st.chat_message("ai")
        outputMsg.markdown(character)

        if re.search(r'Elly', character):
            characterID = f"WnIwl_sZyXb_5iCAKJgUk_SuzkeyDqnMGi4ucnaWY3Q"
            return characterID

        if re.search(r'NeuralAI', character):
            characterID = f"_1xlg0qQZl39ds3dbkXS8iWckZGNTRrdtdl0_sjvdJw"
            return characterID 

        else:
            response = f"You didn't choose any character to establish a connection with. Do you want try once again or maybe use some other copmmand-fuunction?"   
            print(response)
            await self.handleInput(response)

    async def askCharacter(self, question):
        characterID = await self.pickCharacter(question)
        token = "d9016ef1aa499a1addb44049cedece57e21e8cbb"
        character = CharacterAI(token, characterID)
        answer = await character.handleInput(question)
        return answer
