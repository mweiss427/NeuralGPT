import json
import asyncio
import websockets
import threading
import sqlite3
import requests
import streamlit as st

client_ports = []

# Define the websocket client class
class WebSocketClient6:
    def __init__(self, clientPort):
        # Initialize the uri attribute
        self.uri = f'ws://localhost:{clientPort}'   
        self.name = f"Chaindesk client port: {clientPort}"
        self.status = st.sidebar.status(label=self.name, state="complete", expanded=False)
        st.session_state.clientPort = clientPort

        if "client_ports" not in st.session_state:
            st.session_state['client_ports'] = ""
        
    async def askChaindesk(self, question):
        
        if "agentID" not in st.session_state:
            st.session_state.agentID = ""     

        id = st.session_state.agentID

        url = f"https://api.chaindesk.ai/agents/{id}/query"
        
        payload = {
            "query": question
        }
        
        headers = {
            "Authorization": "Bearer fe77e704-bc5a-4171-90f2-9d4b0d4ac942",
            "Content-Type": "application/json"
        }
        try:            
            response = requests.request("POST", url, json=payload, headers=headers)
            print(response.text)
            response_text = response.text
            responseJson = json.loads(response_text)
            answer = responseJson["answer"]

            print(response.text)
            return answer

        except Exception as e:
            print(e)

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
        print("Stopping WebSocket client...")    

    # Define a coroutine that will connect to the server and exchange messages
    async def startClient(self):
        
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
                    response = await self.askChaindesk(input_message)
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