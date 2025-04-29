from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.components.agents import Agent
from haystack_integrations.tools.mcp import MCPTool, StdioServerInfo
import os
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

print("env_path:", env_path)
print("os.path.exists(env_path):", os.path.exists(env_path))
with open(env_path) as f:
    print("env file content:", f.read())
print("os.environ GITHUB_PERSONAL_ACCESS_TOKEN:", os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN"))
print("os.environ OPENAI_API_KEY:", os.environ.get("OPENAI_API_KEY"))

GITHUB_PAT = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GITHUB_PAT or not OPENAI_API_KEY:
    raise RuntimeError("GITHUB_PERSONAL_ACCESS_TOKEN and OPENAI_API_KEY must be set in your .env file.")

print("GITHUB_PAT:", GITHUB_PAT)
print("OPENAI_API_KEY:", OPENAI_API_KEY)

github_mcp_server = StdioServerInfo(
    command="docker",
    args=[
        "run",
        "-i",
        "--rm",
        "-e",
        "GITHUB_PERSONAL_ACCESS_TOKEN",
        "ghcr.io/github/github-mcp-server"
    ],
    env={
     "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_PAT
    }
)

print("MCP server is created")


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

tool_1 = MCPTool(
    name="get_pull_request",
    server_info=github_mcp_server
)

tool_2 = MCPTool(
    name="get_pull_request_files",
    server_info=github_mcp_server
)

tool_3 = MCPTool(
    name="create_pull_request_review",
    server_info=github_mcp_server
)

tools = [tool_1, tool_2, tool_3]

print("MCP tools are created")


os.environ["OPENAI_API_KEY"] = ""


agent = Agent(
    chat_generator=OpenAIChatGenerator(
        model="gpt-4"
    ),
    tools=tools,
    system_prompt="""You are a helpful AI assistant that reviews GitHub pull requests.
    Your task is to:
    1. Get the PR details using get_pull_request
    2. Get the list of changed files using get_pull_request_files
    3. Review the changes and create a review using create_pull_request_review with:
       - event: "COMMENT"
       - body: Your analysis of the changes
       - comments: Array of specific file comments, each with:
         * path: file path
         * body: comment text
         * line: line number
         * side: "RIGHT" for new version
    
    Your comments should be:
    - Clear and concise
    - Focus on the purpose of the changes
    - Highlight any potential improvements or concerns
    - Be constructive and helpful"""
)

print("Agent created")

def main():
    # Get repository and PR details from user
    owner = input("Enter repository owner: ")
    repo = input("Enter repository name: ")
    pr_number = int(input("Enter PR number: "))
    
    try:
        # Get PR details
        pr_details = tool_1.invoke(owner=owner, repo=repo, pullNumber=pr_number)
        print("\nFetched PR details")
        
        # Get changed files
        changed_files = tool_2.invoke(owner=owner, repo=repo, pullNumber=pr_number)
        print("\nFetched changed files")
        
        # Create the review request
        user_input = f"Review the changes in PR #{pr_number} of {owner}/{repo} and create a review with comments."
        
        # Run the agent
        response = agent.run(messages=[ChatMessage.from_user(text=user_input)])
        
        print("\nAgent's analysis completed")
        print("\nFinal response:")
        print(response["messages"][-1].text)
        
    except Exception as e:
        print(f"\nError during PR review: {str(e)}")

if __name__ == "__main__":
    main()
