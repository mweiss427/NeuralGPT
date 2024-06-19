import cohere
from cohere.responses.classify import Example

class IntentClassifier:
    
    def _init__(self):
        
        self.kazik = "franek"
       
    async def intent(self, input):

        co = cohere.Client("Ev0v9wwQPa90xDucdHTyFsllXGVHXouakUMObkNb")

        examples = [
            Example("How are you?", "conversation"), 
            Example("Hello!", "conversation"), 
            Example("Can you explain me how it works?", "conversation"), 
            Example("Tell me a joke", "conversation"), 
            Example("Can you start a websocket server?", "websockets"), 
            Example("Connect client to a websocket server", "websockets"), 
            Example("Disconnect a client from server", "websockets"), 
            Example("Establish a connection", "websockets"), 
            Example("At what port connect the client?", "port"), 
            Example("Start server at port 2001", "port"),
            Example("Connect a client at port 3012", "port"), 
            Example("Which ports are busy?", "port"),
            Example("To which port can client connect", "port"),
            Example("What do you know?", "internet search"), 
            Example("What are the most recent news?", "internet search"), 
            Example("I need to know more", "internet search")            
            ]

        response = co.classify(
            model='embed-english-v3.0',
            inputs=input,
            examples=examples)
        
        classifications = response.classifications
        return str(classifications)