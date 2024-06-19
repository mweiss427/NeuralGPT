import os
import g4f
import json
import websockets
import datetime
import asyncio
import sqlite3
import requests
import http.server
import socketserver
import fireworks.client
import streamlit as st
import streamlit.components.v1 as components
from ServG4F import WebSocketServer1
from ServG4F2 import WebSocketServer3
from ServFire import WebSocketServer
from ServChar import WebSocketServer2
from clientG4F import WebSocketClient1
from forefront import ForefrontClient
from clientG4F2 import WebSocketClient3
from ServFlowise import WebSocketServer5
from clientFlowise import WebSocketClient5
from ServForefront import WebSocketServer4
from ServChaindesk import WebSocketServer6
from PyCharacterAI import Client
from clientChaindesk import WebSocketClient6
from clientForefront import WebSocketClient4
from clientFireworks import WebSocketClient
from clientCharacter import WebSocketClient2
from websockets.sync.client import connect

client = Client()

servers = {}
clients = {}
inputs = []
outputs = []
used_ports = []
server_ports = []
client_ports = []

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
fireworks_api_key = os.getenv("FIREWORKS_API_KEY")

# Stop the WebSocket server
async def stop_websockets():    
    global server
    if server:
        # Close all connections gracefully
        await server.close()
        # Wait for the server to close
        await server.wait_closed()
        print("Stopping WebSocket server...")
    else:
        print("WebSocket server is not running.")

# Stop the WebSocket client
async def stop_client():
    global ws
    # Close the connection with the server
    await ws.close()
    print("Stopping WebSocket client...")

async def main():
       
    st.set_page_config(layout="wide")
    st.title("serverovnia")        

    if "server_ports" not in st.session_state:
        st.session_state['server_ports'] = ""
    if "client_ports" not in st.session_state:
        st.session_state['client_ports'] = ""
    if "user_ID" not in st.session_state:
        st.session_state.user_ID = ""
    if "gradio_Port" not in st.session_state:
        st.session_state.gradio_Port = "" 
    if "server" not in st.session_state:
        st.session_state.server = False
    if "client" not in st.session_state:
        st.session_state.client = False
    if "server_ports" not in st.session_state:
        st.session_state['server_ports'] = ""
    if "client_ports" not in st.session_state:
        st.session_state['client_ports'] = ""
    if "gradio_Port" not in st.session_state:
        st.session_state.gradio_Port = "" 
    if "servers" not in st.session_state:
        st.session_state.servers = None
    if "server" not in st.session_state:
        st.session_state.server = False    
    if "client" not in st.session_state:
        st.session_state.client = False    
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "forefront_api" not in st.session_state:
        st.session_state.forefront_api = ""    
    if "tokenChar" not in st.session_state:
        st.session_state.tokenChar = ""           
    if "charName" not in st.session_state:
        st.session_state.charName = ""
    if "character_ID" not in st.session_state:
        st.session_state.character_ID = "" 
    if "flow" not in st.session_state:
        st.session_state.flow = ""        
    if "agentID" not in st.session_state:
        st.session_state.agentID = "" 
    if "googleAPI" not in st.session_state:
        st.session_state.googleAPI = ""        
    if "cseID" not in st.session_state:
        st.session_state.cseID = ""                        
        
    if "http_server" not in st.session_state:
        
        PORT = 8001
        Handler = http.server.SimpleHTTPRequestHandler
        st.session_state.http_server = True

        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print("serving at port", PORT)
            httpd.serve_forever()    


    userInput = st.chat_input("Ask agent")    

    selectServ = st.selectbox("Select source", ("Fireworks", "Bing", "GPT-3,5", "character.ai", "Forefront", "AgentsGPT", "ChainDesk", "Flowise", "DocsBot"))

    c1, c2 = st.columns(2)
        
    with c1:
        websocketPort = st.number_input("Websocket server port", min_value=1000, max_value=9999, value=1000)   
        startServer = st.button("Start server")
        stopServer = st.button("Stop server")
        st.text("Server ports")
        serverPorts1 = st.container(border=True)
        serverPorts1.markdown(st.session_state['server_ports'])
    
    with c2:
        clientPort = st.number_input("Websocket client port", min_value=1000, max_value=9999, value=1000)
        runClient = st.button("Start client")
        stopClient = st.button("Stop client")        
        st.text("Client ports")
        clientPorts1 = st.container(border=True)
        clientPorts1.markdown(st.session_state['client_ports'])

    with st.sidebar:
        # Wyświetlanie danych, które mogą być modyfikowane na różnych stronach
        serverPorts = st.container(border=True)
        serverPorts.markdown(st.session_state['server_ports'])
        st.text("Client ports")
        clientPorts = st.container(border=True)
        clientPorts.markdown(st.session_state['client_ports'])
        st.text("Character.ai ID")
        user_id = st.container(border=True)
        user_id.markdown(st.session_state.user_ID)    
        status = st.status(label="runs", state="complete", expanded=False)

        if st.session_state.server == True:
            st.markdown("server running...")
        
        if st.session_state.client == True:    
            st.markdown("client running")    
   
    if stopServer:
        stop_websockets

    if stopClient:
        stop_client  

    if selectServ == "Fireworks":
        fireworksAPI = st.text_input("Fireworks API") 

        if startServer:
            fireworks.client.api_key = fireworksAPI
            st.session_state.api_key = fireworks.client.api_key
            server_ports.append(websocketPort)
            st.session_state.server = True
            st.session_state['server_ports'] = server_ports
            serverPorts1.markdown(st.session_state['server_ports'])
            try:
                server = WebSocketServer("localhost")
                servers.append(server)
                print(f"Starting WebSocket server on port {websocketPort}...")
                await server.start_server(websocketPort)
                status.update(label="runs", state="running", expanded=True)
                await asyncio.Future()

            except Exception as e:
                print(f"Error: {e}")

        if runClient:
            st.session_state.client = True
            fireworks.client.api_key = fireworksAPI
            st.session_state.api_key = fireworks.client.api_key
            client_ports.append(clientPort)
            st.session_state['client_ports'] = client_ports
            clientPorts1.markdown(st.session_state['client_ports'])
            try:
                uri = f'ws://localhost:{clientPort}'
                client = WebSocketClient(uri)    
                print(f"Connecting client on port {clientPort}...")
                await client.startClient()
                st.session_state.client = client
                status.update(label="runs", state="running", expanded=True)
                await asyncio.Future()

            except Exception as e:
                print(f"Error: {e}")                

        if userInput:        
            print(f"User B: {userInput}")
            st.session_state.api_key = fireworksAPI
            user_input = st.chat_message("human")
            user_input.markdown(userInput)
            fireworks1 = WebSocketServer()
            response1 = await fireworks1.chatCompletion(userInput)
            print(response1)
            outputMsg = st.chat_message("ai") 
            outputMsg.markdown(response1)

    if selectServ == "Bing":

        if startServer:            
            server_ports.append(websocketPort)
            st.session_state.server = True
            st.session_state['server_ports'] = server_ports
            serverPorts1.markdown(st.session_state['server_ports'])
            try:      
                server1 = WebSocketServer1("localhost", websocketPort)    
                print(f"Starting WebSocket server on port {websocketPort}...")
                await server1.start_server()
                st.session_state.server = server1
                status.update(label="runs", state="running", expanded=True)
                await asyncio.Future()

            except Exception as e:
                print(f"Error: {e}")

        if runClient:
            st.session_state.client = True
            client_ports.append(clientPort)
            st.session_state['client_ports'] = client_ports
            clientPorts1.markdown(st.session_state['client_ports'])
            try:
                client1 = WebSocketClient1(clientPort)    
                print(f"Connecting client on port {clientPort}...")
                await client1.startClient()
                st.session_state.client = client1
                status.update(label="runs", state="running", expanded=True)
                await asyncio.Future()

            except Exception as e:
                print(f"Error: {e}")                

        if userInput:
            user_input1 = st.chat_message("human")
            user_input1.markdown(userInput)
            bing = WebSocketServer1("localhost", websocketPort)
            response = await bing.askQuestion(userInput)
            outputMsg1 = st.chat_message("ai") 
            outputMsg1.markdown(response)

    if selectServ == "GPT-3,5":
        
        if startServer:
            server_ports.append(websocketPort)
            st.session_state.server = True
            st.session_state['server_ports'] = server_ports
            serverPorts1.markdown(st.session_state['server_ports'])
            try:      
                server2 = WebSocketServer3("localhost", websocketPort)    
                print(f"Starting WebSocket server on port {websocketPort}...")
                await server2.start_server()
                st.session_state.server = server2
                await asyncio.Future()

            except Exception as e:
                print(f"Error: {e}")

        if runClient:
            st.session_state.client = True
            client_ports.append(clientPort)
            st.session_state['client_ports'] = client_ports
            clientPorts1.markdown(st.session_state['client_ports'])
            try:
                client2 = WebSocketClient3(clientPort)    
                print(f"Connecting client on port {clientPort}...")
                await client2.startClient()
                st.session_state.client = client2
                await asyncio.Future()

            except Exception as e:
                print(f"Error: {e}")                

        if userInput:
            user_input2 = st.chat_message("human")
            user_input2.markdown(userInput)
            GPT = WebSocketServer3("localhost", websocketPort)
            response = await GPT.askQuestion(userInput)
            outputMsg2 = st.chat_message("ai") 
            outputMsg2.markdown(response)

    if selectServ == "character.ai":

        characterToken = st.text_input("Character AI user token") 
        characterID = st.text_input("Your characters ID") 

        if startServer:
            st.session_state.tokenChar = characterToken
            st.session_state.character_ID = characterID
            server_ports.append(websocketPort)
            st.session_state.server = True
            st.session_state['server_ports'] = server_ports
            serverPorts1.markdown(st.session_state['server_ports'])
            try:
                server = WebSocketServer2(characterToken, characterID)    
                print(f"Starting WebSocket server on port {websocketPort}...")
                await server.start_server(websocketPort)
                status.update(label="runs", state="running", expanded=True)
                await asyncio.Future()

            except Exception as e:
                print(f"Error: {e}")

        if runClient:
            st.session_state.tokenChar = characterToken
            st.session_state.character_ID = characterID
            client_ports.append(clientPort)
            st.session_state['client_ports'] = client_ports
            clientPorts1.markdown(st.session_state['client_ports'])
            try:
                uri = f'ws://localhost:{clientPort}'
                client = WebSocketClient2(clientPort)    
                print(f"Connecting client on port {clientPort}...")
                await client.startClient()
                st.session_state.client = client
                status.update(label="runs", state="running", expanded=True)
                await asyncio.Future()

            except Exception as e:
                print(f"Error: {e}")                

        if userInput:
            print(f"User B: {userInput}")
            character = WebSocketServer2(characterToken, characterID)
            response1 = await character.askCharacter(userInput)
            print(response1)
            return response1

    if selectServ == "ChainDesk":

        agentID = st.text_input("Agent ID")

        if userInput:
            st.session_state.agentID = agentID
            user_input6 = st.chat_message("human")
            user_input6.markdown(userInput)
            chaindesk = WebSocketServer6("localhost")
            response6 = await chaindesk.askChaindesk(userInput)
            outputMsg = st.chat_message("ai")
            outputMsg.markdown(response6)

        if startServer:
            st.session_state.agentID = agentID
            server_ports.append(websocketPort)
            st.session_state.server = True
            st.session_state['server_ports'] = server_ports
            serverPorts1.markdown(st.session_state['server_ports'])
            try:      
                server6 = WebSocketServer6("localhost")    
                print(f"Starting WebSocket server on port {websocketPort}...")
                await server6.start_server(websocketPort)
                st.session_state.server = server6
                await asyncio.Future()

            except Exception as e:
                print(f"Error: {e}")

        if runClient:
            st.session_state.agentID = agentID
            st.session_state.client = True
            client_ports.append(clientPort)
            st.session_state['client_ports'] = client_ports
            clientPorts1.markdown(st.session_state['client_ports'])
            try:
                client6 = WebSocketClient6(clientPort)    
                print(f"Connecting client on port {clientPort}...")
                await client6.startClient()
                st.session_state.client = client6
                await asyncio.Future()

            except Exception as e:
                print(f"Error: {e}")  

    if selectServ == "Flowise":
        
        flow = st.text_input("flow ID")

        if userInput:
            st.session_state.flow = flow
            user_input6 = st.chat_message("human")
            user_input6.markdown(userInput)
            flowise = WebSocketServer5("localhost")
            response6 = await flowise.askQuestion(userInput)
            outputMsg = st.chat_message("ai")
            outputMsg.markdown(response6)

        if startServer:
            st.session_state.flow = flow
            server_ports.append(websocketPort)
            st.session_state.server = True
            st.session_state['server_ports'] = server_ports
            serverPorts1.markdown(st.session_state['server_ports'])
            try:      
                server5 = WebSocketServer5("localhost")    
                print(f"Starting WebSocket server on port {websocketPort}...")
                await server5.start_server()
                st.session_state.server = server5
                await asyncio.Future()

            except Exception as e:
                print(f"Error: {e}")

        if runClient:
            st.session_state.flow = flow
            st.session_state.client = True
            client_ports.append(clientPort)
            st.session_state['client_ports'] = client_ports
            clientPorts1.markdown(st.session_state['client_ports'])
            try:
                client5 = WebSocketClient5(clientPort)    
                print(f"Connecting client on port {clientPort}...")
                await client5.startClient(clientPort)
                st.session_state.client = client5
                await asyncio.Future()

            except Exception as e:
                print(f"Error: {e}")                          

    if selectServ == "DocsBot":

        botID = st.text("Docsbot ID")

        url = f"http://localhost:8001/Docsbotport.html"
        st.components.v1.iframe(url, height=950, scrolling=True)

asyncio.run(main())            