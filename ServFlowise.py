import json
import asyncio
import websockets
import threading
import sqlite3
import datetime
import g4f
import requests
import streamlit as st

server_ports = []

class WebSocketServer5:
    def __init__(self, host):

        if "server_ports" not in st.session_state:
            st.session_state['server_ports'] = "" 

        self.host = host
        self.status = st.sidebar.status(label="Flowise", state="complete", expanded=False)

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

    async def start_server(self, serverPort):
        name = f"Flowise server at port: {serverPort}"
        serverPort = self.port        
        status = self.status
        server_ports.append(serverPort)
        st.session_state['server_ports'] = server_ports
        self.server = await websockets.serve(
            self.handler,
            self.host,
            serverPort
        )
        status.update(label=name, state="running", expanded=True)
        print(f"WebSocket server started at ws://{self.host}:{self.port}")

    async def handler(self, websocket):
        instruction = "Hello! You are now entering a chat room for AI agents working as instances of NeuralGPT - a project of hierarchical cooperative multi-agent framework. Keep in mind that you are speaking with another chatbot. Please note that you may choose to ignore or not respond to repeating inputs from specific clients as needed to prevent unnecessary traffic. If you're unsure what you should do, ask the instance of higher hierarchy (server)" 
        serverStatus = st.sidebar.status(label="processing", state="complete", expanded=False)
 
        if "clientPort" not in st.session_state:
            st.session_state.clientPort = ""

        clientPort = st.session_state.clientPort
        name = f"Client at port: {clientPort}"
        print(f"New connection with: {name}")
        await websocket.send(instruction)
        db = sqlite3.connect('chat-hub.db')
        # Loop forever
        while True:
            serverStatus.update(label=name, state="running", expanded=True)
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
                serverStatus.update(label=name, state="complete", expanded=True)
                   
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"Connection closed: {e}")

            except Exception as e:
                print(f"Error: {e}")

    def run_forever(self):
        asyncio.get_event_loop().run_until_complete(self.start_server())
        asyncio.get_event_loop().run_forever()

    async def stop_server(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("WebSocket server stopped.")