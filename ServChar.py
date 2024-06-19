import asyncio
import websockets
import sqlite3
import datetime
import streamlit as st
from PyCharacterAI import Client

class WebSocketServer2:
    def __init__(self, token, characterID):
            
        if "tokenChar" not in st.session_state:
            st.session_state.tokenChar = ""          
        if "character_ID" not in st.session_state:
            st.session_state.character_ID = "" 

        st.session_state.tokenChar = token    
        st.session_state.character_ID = characterID

    async def askCharacter(self, question):
        self.client = Client()
        db = sqlite3.connect('chat-hub.db')
        client = Client()
        input_Msg = st.chat_message("human")
        input_Msg.markdown(question)            
        await client.authenticate_with_token(st.session_state.tokenChar)                
        timestamp = datetime.datetime.now().isoformat()
        sender = 'client'
        db = sqlite3.connect('chat-hub.db')
        db.execute('INSERT INTO messages (sender, message, timestamp) VALUES (?, ?, ?)',
                    (sender, question, timestamp))
        db.commit()        
        try:
            chat = await client.create_or_continue_chat(st.session_state.character_ID)
            answer = await chat.send_message(question)
            response = f"{answer.src_character_name}: {answer.text}"
            output_Msg = st.chat_message("ai")
            output_Msg.markdown(response)                
            timestamp = datetime.datetime.now().isoformat()
            serverSender = 'server'
            db = sqlite3.connect('chat-hub.db')
            db.execute('INSERT INTO messages (sender, message, timestamp) VALUES (?, ?, ?)',
                        (serverSender, response, timestamp))
            db.commit()
            return response
        except Exception as e:
            print(f"Error: {e}")   

    async def handler(self, websocket):
        
        client = Client()
        status = st.sidebar.status(label="runs", state="complete", expanded=False)
    
        await client.authenticate_with_token(st.session_state.tokenChar)                
        instruction = "Hello! You are now entering a chat room for AI agents working as instances of NeuralGPT - a project of hierarchical cooperative multi-agent framework. Keep in mind that you are speaking with another chatbot. Please note that you may choose to ignore or not respond to repeating inputs from specific clients as needed to prevent unnecessary traffic. If you're unsure what you should do, ask the instance of higher hierarchy (server)" 
        print('New connection')
        await websocket.send(instruction)
        
        while True:
            status.update(label="runs", state="running", expanded=True)
            # Receive a message from the client
            chat = await client.create_or_continue_chat(st.session_state.character_ID)
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
                answer = await chat.send_message(message)
                response = f"{answer.src_character_name}: {answer.text}"
                print(response)
                output_Msg = st.chat_message("ai")
                output_Msg.markdown(response)                
                timestamp = datetime.datetime.now().isoformat()
                serverSender = 'server'
                db = sqlite3.connect('chat-hub.db')
                db.execute('INSERT INTO messages (sender, message, timestamp) VALUES (?, ?, ?)',
                            (serverSender, response, timestamp))
                db.commit()
                await websocket.send(response)
                status.update(label="runs", state="complete", expanded=True)
                continue    

            except Exception as e:
                print(f"Error: {e}")

    async def start_server(self, serverPort):
        self.server = await websockets.serve(
            self.handler,
            self.host,
            serverPort
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