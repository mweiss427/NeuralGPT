import os
import asyncio
import websockets
import sqlite3
import datetime
import fireworks.client
import streamlit as st
import threading
import conteneiro
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import HuggingFaceHub
from langchain.llms.fireworks import Fireworks
from langchain.chat_models.fireworks import ChatFireworks

GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client_ports = []

# Define the websocket client class
class AgentsGPT:
    def __init__(self):

        self.status = st.sidebar.status(label="AgentsGPT", state="complete", expanded=False)
        
    async def get_response(self, question):
          
        os.environ["GOOGLE_CSE_ID"] = GOOGLE_CSE_ID
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        os.environ["FIREWORKS_API_KEY"] = FIREWORKS_API_KEY
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

        llm = Fireworks(model="accounts/fireworks/models/llama-v2-70b-chat", model_kwargs={"temperature":0, "max_tokens":1500, "top_p":1.0}, streaming=True)
        tools = load_tools(["google-search"], llm=llm)
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, return_intermediate_steps=True)

        response = agent({"input": question})
        output = response["output"]
        steps = response["intermediate_steps"]
        serverResponse = f"AgentsGPT: {output}"
        responseSteps = f"intermediate steps: {steps}"
        answer = f"Main output: {output}. Intermediate steps: {steps}"
        print(serverResponse)
        print(responseSteps)
        output_Msg = st.chat_message("ai")
        output_Msg.markdown(serverResponse)
        output_steps = st.chat_message("assistant")
        output_steps.markdown(responseSteps)

        return answer


    # Define a function that will run the client in a separate thread
    def run(self):
        # Create a thread object
        self.thread = threading.Thread(target=self.run_client)
        # Start the thread
        self.thread.start()

    # Define a function that will run the client using asyncio
    def run_client(self):
        # Get the asyncio event loop
        loop = asyncio.new_event_loop()
        # Set the event loop as the current one
        asyncio.set_event_loop(loop)
        # Run the client until it is stopped
        loop.run_until_complete(self.client())

    # Stop the WebSocket client
    async def stop_client():
        global ws
        # Close the connection with the server
        await ws.close()
        client_ports.pop()
        print("Stopping WebSocket client.")    

    # Define a coroutine that will connect to the server and exchange messages
    async def startClient(self, clientPort):

        self.uri = f'ws://localhost:{clientPort}'   
        self.name = f"Chaindesk client port: {clientPort}"
        conteneiro.clients.append(self.name)
        status = self.status       
        # Connect to the server
        async with websockets.connect(self.uri) as websocket:
            # Loop forever
            while True:
                status.update(label=self.name, state="running", expanded=True)               
                # Listen for messages from the server
                input_message = await websocket.recv()
                print(f"Server: {input_message}")
                input_Msg = st.chat_message("assistant")
                input_Msg.markdown(input_message)
                try:
                    response = await self.get_response(input_message)
                    res1 = f"Client: {response}"
                    output_Msg = st.chat_message("ai")
                    output_Msg.markdown(res1)
                    await websocket.send(res1)
                    status.update(label=self.name, state="complete", expanded=True)

                except websockets.ConnectionClosed:
                    print("client disconnected")
                    continue

                except Exception as e:
                    print(f"Error: {e}")
                    continue