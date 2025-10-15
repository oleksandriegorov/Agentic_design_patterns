import os, getpass
import asyncio
import nest_asyncio
from typing import List
from dotenv import load_dotenv
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool as langchain_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

# UNCOMMENT
# Prompt the user securely and set API keys as an environment variables
# os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")
# os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

try:
    # A model with function/tool calling capabilities is required.
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    print(f"✅ Language model initialized: {llm.model}")
except Exception as e:
    print(f"🛑 Error initializing language model: {e}")
    llm = None

# Original simulated results dictionary
simulated_results = {
    "weather in london": "The weather in London is currently cloudy with a temperature of 15°C.",
    "capital of france": "The capital of France is Paris.",
    "population of earth": "The estimated population of Earth is around 8 billion people.",
    "tallest mountain": "Mount Everest is the tallest mountain above sea level.",
    "default": "Simulated search result for '{query}': No specific information found, but the topic seems interesting."
}

# --- Define 5 Specific Tools ---
@langchain_tool
def get_weather_info(city: str) -> str:
    """
    Provides current weather information for a specified city.
    Use this tool for queries like 'weather in London'.
    """
    query_key = f"weather in {city.lower()}"
    print(f"\n--- 🛠️ Tool Called: get_weather_info for city: '{city}' ---")
    result = simulated_results.get(query_key, simulated_results["default"].format(query=query_key))
    print(f"--- TOOL RESULT: {result} ---")
    return result

@langchain_tool
def get_capital_info(country: str) -> str:
    """
    Provides the capital city of a specified country.
    Use this tool for queries like 'capital of France'.
    """
    query_key = f"capital of {country.lower()}"
    print(f"\n--- 🛠️ Tool Called: get_capital_info for country: '{country}' ---")
    result = simulated_results.get(query_key, simulated_results["default"].format(query=query_key))
    print(f"--- TOOL RESULT: {result} ---")
    return result

@langchain_tool
def get_population_info(entity: str) -> str:
    """
    Provides population information for a specified entity (e.g., 'Earth').
    Use this tool for queries like 'population of Earth'.
    """
    query_key = f"population of {entity.lower()}"
    print(f"\n--- 🛠️ Tool Called: get_population_info for entity: '{entity}' ---")
    result = simulated_results.get(query_key, simulated_results["default"].format(query=query_key))
    print(f"--- TOOL RESULT: {result} ---")
    return result

@langchain_tool
def get_mountain_info(mountain_name: str) -> str:
    """
    Provides information about a specified mountain.
    Use this tool for queries like 'tallest mountain'.
    """
    query_key = f"{mountain_name.lower()}"
    print(f"\n--- 🛠️ Tool Called: get_mountain_info for mountain: '{mountain_name}' ---")
    result = simulated_results.get(query_key, simulated_results["default"].format(query=query_key))
    print(f"--- TOOL RESULT: {result} ---")
    return result

@langchain_tool
def get_general_info(query: str) -> str:
    """
    Provides general factual information on a given topic when no other specific tool applies.
    This acts as a fallback for broader queries.
    """
    print(f"\n--- 🛠️ Tool Called: get_general_info with query: '{query}' ---")
    # This tool will always return the default message as it's a general fallback
    result = simulated_results["default"].format(query=query)
    print(f"--- TOOL RESULT: {result} ---")
    return result


tools = [get_weather_info, get_capital_info, get_population_info, get_mountain_info, get_general_info]

# --- Create a Tool-Calling Agent ---
if llm:
    # This prompt template requires an `agent_scratchpad` placeholder for the agent's internal steps.
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the most appropriate tool to answer the user's query."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Create the agent, binding the LLM, tools, and prompt together.
    agent = create_tool_calling_agent(llm, tools, agent_prompt)

    # AgentExecutor is the runtime that invokes the agent and executes the chosen tools.
    # The 'tools' argument is not needed here as they are already bound to the agent.
    agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools, return_intermediate_steps=True)

async def run_agent_with_tool(query: str):
    """Invokes the agent executor with a query and prints the final response."""
    print(f"\n--- 🏃 Running Agent with Query: '{query}' ---")
    try:
        response = await agent_executor.ainvoke({"input": query})
        print("\n--- Intermediate Steps ---")
        for step in response.get("intermediate_steps", []):
            print(step)
        print("\n--- ✅ Final Agent Response ---")
        print(response["output"])
    except Exception as e:
        print(f"\n🛑 An error occurred during agent execution: {e}")

async def main():
    """Runs all agent queries concurrently."""
    tasks = [
        run_agent_with_tool("What is the capital of France?"),
        run_agent_with_tool("What's the weather like in London?"),
        run_agent_with_tool("What is the population of Earth?"),
        run_agent_with_tool("Tell me about the tallest mountain."),
        run_agent_with_tool("Tell me something about dogs."), # Should trigger the general info tool
        run_agent_with_tool("Who is the president of the USA?") # Also general info
    ]
    await asyncio.gather(*tasks)

nest_asyncio.apply()
asyncio.run(main())