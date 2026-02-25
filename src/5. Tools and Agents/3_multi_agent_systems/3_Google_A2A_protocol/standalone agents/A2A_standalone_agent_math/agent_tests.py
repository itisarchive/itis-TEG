from dotenv import load_dotenv

from agent import Agent

load_dotenv()

agent = Agent()
sessionID = "session_12345"

query = "How much is 100 / 4?"
response = agent.invoke(query, sessionID)
print(response)
