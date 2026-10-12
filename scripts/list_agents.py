"""One-off: list every agent in the project with its real ID, since the
portal's GUID isn't the same as the API-facing agent ID."""
import os
from azure.ai.agents import AgentsClient
from azure.identity import InteractiveBrowserCredential
from dotenv import load_dotenv

load_dotenv()

client = AgentsClient(
    endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
    credential=InteractiveBrowserCredential(tenant_id=os.environ.get("AZURE_TENANT_ID")),
)

for agent in client.list_agents():
    print(f"{agent.id}  —  {agent.name}")
