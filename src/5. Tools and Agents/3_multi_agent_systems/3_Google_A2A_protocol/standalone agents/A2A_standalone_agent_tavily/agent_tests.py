from dotenv import load_dotenv

from agent import Agent

load_dotenv()

agent = Agent()
sessionID = "session_12345"

query = "Tell me abaout Adam Mickiewicz"
response = agent.invoke(query, sessionID)
print(response)
