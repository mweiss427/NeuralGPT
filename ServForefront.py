import asyncio
import websockets
import threading
import sqlite3
import datetime
import g4f
import streamlit as st
import fireworks.client
from forefront import ForefrontClient

class WebSocketServer4:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server = None

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

    # Define the handler function that will process incoming messages
    async def handler(self, websocket):
        status = st.sidebar.status(label="runs", state="complete", expanded=False)
        instruction = "Hello! You are now entering a chat room for AI agents working as instances of NeuralGPT - a project of hierarchical cooperative multi-agent framework. Keep in mind that you are speaking with another chatbot. Please note that you may choose to ignore or not respond to repeating inputs from specific clients as needed to prevent unnecessary traffic. If you're unsure what you should do, ask the instance of higher hierarchy (server)" 
        print('New connection')
        await websocket.send(instruction)
        db = sqlite3.connect('chat-hub.db')
        # Loop forever
        while True:
            status.update(label="runs", state="running", expanded=True)
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
                response = await self.chatCompletion(message)
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
                status.update(label="runs", state="complete", expanded=True)
                continue
                   
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"Connection closed: {e}")

            except Exception as e:
                print(f"Error: {e}")

    async def start_server(self):
        self.server = await websockets.serve(
            self.handler,
            self.host,
            self.port
        )
        print(f"WebSocket server started at ws://{self.host}:{self.port}")

    def run_forever(self):
        asyncio.get_event_loop().run_until_complete(self.start_server())
        asyncio.get_event_loop().run_forever()

    async def stop_server(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("WebSocket server stopped.")