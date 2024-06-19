import json
import datetime
import sqlite3
import asyncio
import requests
import conteneiro
import websockets
import streamlit as st
from agentLangchain import langchainAgent

servers = []
clients = []
inputs = []
outputs = []
messagess = []
intentios = []
used_ports = []
server_ports = []
client_ports = []

db = sqlite3.connect('chat-hub.db')
cursor = db.cursor()
cursor.execute('CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, sender TEXT, message TEXT, timestamp TEXT)')    
db.commit()

async def main():

    userInput = st.chat_input("Ask Agent")         

    c1, c2 = st.columns(2)
    fireAPI = st.text_input("Fireworks API")

    with c1:
        stat1 = st.empty()
        state1 = stat1.status(label="Langchain", state="complete", expanded=False)
        state1.write(conteneiro.servers)
        websocketPort = st.number_input("Websocket servers", min_value=1000, max_value=9999, value=1000)   
        startServer = st.button("Start server")
        stopServer = st.button("Stop server")            

    with c2:
        stat2 = st.empty()
        state2 = stat2.status(label="Langchain", state="complete", expanded=False)
        state2.write(conteneiro.clients)
        clientPort = st.number_input("Websocket clients", min_value=1000, max_value=9999, value=1000)   
        start_Client = st.button("Start client")
        stopClient = st.button("Stop client")       

    with st.sidebar:
        cont = st.empty()        
        status = cont.status(label="Langchain", state="complete", expanded=False)
        

    if userInput:
        user_input = st.chat_message("human")
        user_input.markdown(userInput)
        messagess.append(user_input)
        agent = langchainAgent(fireAPI)
        response = await agent.askQuestion(userInput)
        outputMsg = st.chat_message("ai") 
        outputMsg.markdown(response)
        await agent.handleInput(response)
        
    if start_Client:
        voiceCli = f"Langchain client port: {clientPort}"
        stat2.empty()            
        state2 = stat2.status(label=voiceCli, state="running", expanded=True)
        state2.write(conteneiro.clients)
        cont.empty()            
        status = cont.status(label=voiceCli, state="running", expanded=True)
        status.write(conteneiro.servers) 
        try:
            client = langchainAgent(fireAPI)
            await client.startClient(clientPort)
            print(f"Connecting client on port {clientPort}...")
            await asyncio.Future()

        except Exception as e:
            print(f"Error: {e}")   

    if startServer:
        vooiceSrv = f"Langchain server pport: {websocketPort}"
        stat1.empty()
        state1 = stat1.status(label=vooiceSrv, state="running", expanded=False)
        state1.write(conteneiro.clients)
        cont.empty()            
        status = cont.status(label=vooiceSrv, state="running", expanded=False)
        status.write(conteneiro.clients)          
        try:          
            server = langchainAgent(fireAPI)
            await server.start_server(websocketPort)                
            print(f"Starting WebSocket server on port {websocketPort}...")
            await asyncio.Future()

        except Exception as e:
            print(f"Error: {e}")  

asyncio.run(main())            