from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.tools.mcp import MCPTool, StdioServerInfo
from haystack.components.agents import Agent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

GITHUB_PAT = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GITHUB_PAT or not OPENAI_API_KEY:
    raise RuntimeError("GITHUB_PERSONAL_ACCESS_TOKEN and OPENAI_API_KEY must be set in your .env file.")


# MCP server setup
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

# MCP tools
def get_mcp_tools():
    file_tool = MCPTool(
        name="get_file_contents",
        server_info=github_mcp_server
    )
    search_code_tool = MCPTool(
        name="search_code",
        server_info=github_mcp_server
    )
    create_or_update_file_tool = MCPTool(
        name="create_or_update_file",
        server_info=github_mcp_server
    )
    return [file_tool, search_code_tool, create_or_update_file_tool]


system_prompt = """You are an AI assistant that generates a high-quality README.md for a given GitHub repository.\n1. 
List all files in the repository.
\n2. Analyze all project configuration and build files (e.g., Makefile, requirements.txt, setup.py, package.json, pyproject.toml, Pipfile, environment.yml, etc.).
\n3. Generate a README.md in English with the following sections:
  - Project Description
  - Installation Instructions
  - Usage Instructions
Your output should be clear, concise, and beginner-friendly."""

def main():
    
    owner = input("Enter repository owner: ")
    repo = input("Enter repository name: ")
    branch = input("Enter branch name: ")

    try:
        tools = get_mcp_tools()
        agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4"),
            tools=tools,
            system_prompt=system_prompt
        )
        user_input = f"Generate a beginner-friendly README.md for the repository {owner}/{repo} on branch {branch}. Analyze all project configuration and build files."
        response = agent.run(messages=[ChatMessage.from_user(text=user_input)],
                            global_kwargs={"owner": owner, "repo": repo, "branch": branch})
        readme_content = response["messages"][-1].text
        print("\nAgent's README generation completed")
        print("\nGenerated README.md content:\n")
        print(readme_content)

        # Push README.md to GitHub using create_or_update_file tool
        create_or_update_file_tool = tools[2]
        commit_message = "Add or update README.md via AI agent"
        print("\nPushing README.md to GitHub repository...")
        push_result = create_or_update_file_tool.invoke(
            owner=owner,
            repo=repo,
            branch=branch,
            path="README.md",
            content=readme_content,
            message=commit_message
        )
        print("\nREADME.md pushed to GitHub!")
        print(push_result)
    except Exception as e:
        print(f"\nError during README generation: {str(e)}")

if __name__ == "__main__":
    main()

