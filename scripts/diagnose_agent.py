import os
from azure.ai.agents import AgentsClient
from azure.identity import InteractiveBrowserCredential
from dotenv import load_dotenv

load_dotenv()

cred = InteractiveBrowserCredential(tenant_id=os.environ.get("AZURE_TENANT_ID"))
client = AgentsClient(endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"], credential=cred)

print("Endpoint:", os.environ["FOUNDRY_PROJECT_ENDPOINT"])

agents = list(client.list_agents())
print("Agent count via list_agents():", len(agents))
for a in agents:
    print(" -", a.id, a.name)

guid = os.environ["MODEL_STACK_ANALYSIS_AGENT_ID"]
try:
    a = client.get_agent(guid)
    print("get_agent(guid) succeeded:", a.id, a.name)
except Exception as e:
    print("get_agent(guid) failed:", type(e).__name__, str(e)[:300])
