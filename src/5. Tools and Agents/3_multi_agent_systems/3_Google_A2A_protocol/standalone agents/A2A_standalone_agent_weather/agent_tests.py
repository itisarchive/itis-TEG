from dotenv import load_dotenv

from agent import WikipediaAgent

load_dotenv()

agent = WikipediaAgent()
sessionID = "session_12345"

query = "Tell me abaout Adam Mickiewicz"
response = agent.invoke(query, sessionID)
print(response)
