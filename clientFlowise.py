import asyncio
import websockets
import threading
import sqlite3
import g4f
import requests
import streamlit as st

client_ports = []

# Define the websocket client class
class WebSocketClient5:
    def __init__(self):
        
        # Initialize the uri attribute
        if "client_ports" not in st.session_state:
            st.session_state['client_ports'] = ""

    async def askQuestion(self, question):

        if "flow" not in st.session_state:
            st.session_state.flow = ""
        
        flow = st.session_state.flow

        API_URL = f"http://localhost:3000/api/v1/prediction/{flow}"
        
        try:
            def query(payload):
                response = requests.post(API_URL, json=payload)
                return response.json()
                
            response = query({
                "question": question,
            })   

            print(response)
            answer = response["text"]
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
    async def startClient(self, clientPort):
        uri = f'ws://localhost:{clientPort}'
        client_ports.append(clientPort)
        st.session_state['client_ports'] = client_ports
        name = f"Flowise client port: {clientPort}"
        status = st.sidebar.status(label=name, state="complete", expanded=False)       
        # Connect to the server
        async with websockets.connect(uri) as websocket:
            # Loop forever
            while True:
                status.update(label=name, state="running", expanded=True)               
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
                    status.update(label=name, state="complete", expanded=True)
                    continue

                except websockets.ConnectionClosed:
                    print("client disconnected")
                    continue

                except Exception as e:
                    print(f"Error: {e}")
                    continue