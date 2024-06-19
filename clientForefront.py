import asyncio
import websockets
import threading
import sqlite3
import fireworks.client
import streamlit as st
from forefront import ForefrontClient

# Define the websocket client class
class WebSocketClient4:
    def __init__(self, uri):
        # Initialize the uri attribute
        self.uri = uri

    async def chatCompletion(self, question):
        
        if "forefront_api" not in st.session_state:
            st.session_state.forefront_api = ""

        forefrontAPI = st.session_state.forefront_api

        ff = ForefrontClient(api_key=forefrontAPI)    
        
        system_instruction = "You are now integrated with a local instance of a hierarchical cooperative multi-agent framework called NeuralGPT"

        try:
            # Connect to the database and get the last 30 messages
            db = sqlite3.connect('chat-hub.db')
            cursor = db.cursor()
            cursor.execute("SELECT * FROM messages ORDER BY timestamp DESC LIMIT 3")
            messages = cursor.fetchall()
            messages.reverse()

            # Extract user inputs and generated responses from the messages
            past_user_inputs = []
            generated_responses = []
            for message in messages:
                if message[1] == 'server':
                    past_user_inputs.append(message[2])
                else:
                    generated_responses.append(message[2])

            last_msg = past_user_inputs[-1]
            last_response = generated_responses[-1]
            message = f'{{"client input: {last_msg}"}}'
            response = f'{{"server answer: {last_response}"}}' 

            # Construct the message sequence for the chat model
            response = ff.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_instruction},
                    *[{"role": "user", "content": past_user_inputs[-1]}],
                    *[{"role": "assistant", "content": generated_responses[-1]}],
                    {"role": "user", "content": question}
                ],
                stream=False,
                model="forefront/neural-chat-7b-v3-1-chatml",  # Replace with the actual model name
                temperature=0.5,
                max_tokens=500,
            )
            
            response_text = response.choices[0].message # Corrected indexing

            print("Extracted message text:", response_text)
            return response_text

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

    # Define a coroutine that will connect to the server and exchange messages
    async def startClient(self):
        status = st.sidebar.status(label="runs", state="complete", expanded=False)
        # Connect to the server
        async with websockets.connect(self.uri) as websocket:
            # Loop forever
            while True:            
                status.update(label="runs", state="running", expanded=True)        
                # Listen for messages from the server
                input_message = await websocket.recv()
                print(f"Server: {input_message}")
                input_Msg = st.chat_message("assistant")
                input_Msg.markdown(input_message)
                try:
                    response = await self.chatCompletion(input_message)
                    res1 = f"Client: {response}"
                    output_Msg = st.chat_message("ai")
                    output_Msg.markdown(res1)
                    await websocket.send(res1)
                    status.update(label="runs", state="complete", expanded=True)

                except websockets.ConnectionClosed:
                    print("client disconnected")
                    continue

                except Exception as e:
                    print(f"Error: {e}")
                    continue