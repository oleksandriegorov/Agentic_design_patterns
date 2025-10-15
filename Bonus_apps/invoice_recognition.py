# Main application file for the Agentic Invoice Recognition App
# To run this:
# 1. Make sure you have Python 3.9+ installed and Poppler installed on your system.
#    (On Mac: brew install poppler, on Debian/Ubuntu: sudo apt-get install poppler-utils)
# 2. Install necessary libraries:
#    pip install langchain langchain_google_genai Pillow reportlab pdf2image
# 3. Set up your Google API key as an environment variable:
#    export GOOGLE_API_KEY="your_google_api_key"
# 4. Place a sample invoice PDF file named 'sample_invoice.pdf' in the same directory.
# 5. Run the script: python invoice_processor.py

import os
import base64
import io
import json
from typing import List, Dict, Any, Optional

from PIL import Image
from pdf2image import convert_from_path

from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
# --- Configuration ---
PLANNING_MODEL = "gemini-2.0-flash"
VISION_MODEL = "gemini-2.0-flash"
REASONING_MODEL = "gemini-2.5-flash-lite"

# --- Pydantic Models for Structured Output ---


class VisionOutput(BaseModel):
    """Data model for the output of the vision sub-agent."""

    transcribed_text: str = Field(
        description="The full text transcribed from the invoice image."
    )
    layout_description: str = Field(
        description="A brief description of the document's layout (e.g., header, footer, table positions)."
    )


class CompanyNamesOutput(BaseModel):
    """Data model for the output of the reasoning sub-agent."""

    vendor_name: Optional[str] = Field(
        description="The name of the company that sent the invoice."
    )
    customer_name: Optional[str] = Field(
        description="The name of the company that received the invoice."
    )


# --- Helper function to convert image to base64 ---
def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL image to a base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# --- Agent Definitions ---


def create_vision_sub_agent():
    """Creates a sub-agent that analyzes an invoice image and returns JSON."""
    llm = ChatGoogleGenerativeAI(model=VISION_MODEL, temperature=0)
    parser = JsonOutputParser(pydantic_object=VisionOutput)

    prompt = PromptTemplate(
        template="""
        You are an expert document analysis agent.
        Your task is to analyze the provided image of an invoice.
        Transcribe all the text you see and briefly describe the layout.
        Respond with a JSON object that follows the specified format.

        {format_instructions}

        Analyze the following image:
        """,
        input_variables=[],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    def agent_logic(image_base64: str) -> Dict[str, Any]:
        """The logic the vision agent will execute."""
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt.format()},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ]
        )
        chain = llm | parser
        return chain.invoke([message])

    return agent_logic


def create_reasoning_sub_agent():
    """Creates a sub-agent that extracts company names from vision agent's output."""
    llm = ChatGoogleGenerativeAI(model=REASONING_MODEL, temperature=0)
    parser = JsonOutputParser(pydantic_object=CompanyNamesOutput)

    prompt = PromptTemplate(
        template="""
        You are a reasoning agent specializing in entity extraction.
        From the provided JSON data, which contains transcribed text from an invoice,
        identify the vendor company name and the customer company name.
        Return the names in a new JSON object.

        {format_instructions}

        Here is the input JSON from the vision agent:
        {vision_agent_output}
        """,
        input_variables=["vision_agent_output"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    return chain


# --- Main Planner Agent ---


class InvoiceProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.llm = ChatGoogleGenerativeAI(temperature=0, model=PLANNING_MODEL)

        # Instantiate sub-agents
        self.vision_agent_logic = create_vision_sub_agent()
        self.reasoning_agent_chain = create_reasoning_sub_agent()

        self.tools = self._create_tools()
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,
            return_messages=True,
        )

    def _create_tools(self) -> List[Tool]:
        """Creates tools for the planner agent by wrapping sub-agent logic."""

        def run_vision_agent_on_pdf(file_path: str) -> str:
            """
            This tool takes a PDF file path, converts the first page to an image,
            and runs the vision sub-agent on it. It returns a JSON string.
            """
            print(f"--- Vision Agent Tool: Processing '{file_path}' ---")
            if not os.path.exists(file_path):
                return json.dumps({"error": "File not found."})
            try:
                images = convert_from_path(file_path)
                if not images:
                    return json.dumps({"error": "PDF is empty or could not be read."})

                # Use the first page for this example
                image_base64 = image_to_base64(images[0])
                result = self.vision_agent_logic(image_base64)
                print("--- Vision Agent Tool: Successfully analyzed image. ---")
                return json.dumps(result)
            except Exception as e:
                return json.dumps(
                    {"error": f"Failed to process PDF with vision agent: {e}"}
                )

        def run_reasoning_agent(vision_output_str: str) -> str:
            """
            This tool takes the JSON string output from the vision agent
            and runs the reasoning sub-agent on it to extract company names.
            It returns a new JSON string.
            """
            print("--- Reasoning Agent Tool: Processing vision output ---")
            try:
                # The input to the reasoning agent is the raw JSON string
                result = self.reasoning_agent_chain.invoke(
                    {"vision_agent_output": vision_output_str}
                )
                print(
                    "--- Reasoning Agent Tool: Successfully extracted company names. ---"
                )
                return json.dumps(result)
            except Exception as e:
                return json.dumps(
                    {
                        "error": f"Failed to process vision output with reasoning agent: {e}"
                    }
                )

        tools = [
            Tool(
                name="VisionSubAgent",
                func=run_vision_agent_on_pdf,
                description="Use this tool to analyze the invoice PDF. It takes a file path as input and returns a JSON string with the transcribed text and layout description.",
            ),
            Tool(
                name="ReasoningSubAgent",
                func=run_reasoning_agent,
                description="Use this tool to extract company names from the JSON output of the VisionSubAgent. It takes the JSON string as input and returns a new JSON string with vendor and customer names.",
            ),
        ]
        return tools

    def run(self):
        """Executes the invoice processing workflow orchestrated by the planning agent."""
        print("--- Planning Agent: Starting invoice processing workflow. ---")

        prompt = self._create_prompt_template()
        agent = create_react_agent(self.llm, self.tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
        )

        initial_task = f"""
        Process the invoice at '{self.pdf_path}'.
        1. First, use the VisionSubAgent to 'glance' at the document and get its contents in JSON format.
        2. Then, take the JSON string output from the VisionSubAgent and pass it to the ReasoningSubAgent to extract the company names.
        3. Finally, present the extracted company names as the final answer.
        """

        result = agent_executor.invoke({"input": initial_task})

        print("\n--- Workflow Complete ---")
        print("Final Result:")
        print(result["output"])
        return result

    def _create_prompt_template(self):
        return PromptTemplate.from_template(
            """
        You are a master planner orchestrating a team of AI sub-agents to process an invoice.
        Your job is to break down the main goal into a sequence of sub-tasks and delegate them to the correct sub-agent using the available tools.

        You have access to the following tools (which represent your sub-agents):
        {tools}

        Follow this process:
        1.  **Thought:** Analyze the user's request and the current state. Decide which sub-agent to call next.
        2.  **Action:** The name of the tool to use. It must be one of [{tool_names}].
        3.  **Action Input:** The input required for that tool.
        4.  **Observation:** The JSON result returned by the tool.

        Repeat this process, chaining the output of one sub-agent to the input of the next, until the final goal is achieved.

        Begin!

        User's Request: {input}
        Chat History: {chat_history}
        Thought:
        {agent_scratchpad}
        """
        )


# --- Main Execution ---
if __name__ == "__main__":
    sample_pdf_path = "/Users/oleksandriegorov/Documents/AI_Projects/Agentic_design_patterns/sample_invoice.pdf"
    if not os.path.exists(sample_pdf_path):
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter

            c = canvas.Canvas(sample_pdf_path, pagesize=letter)
            width, height = letter
            c.drawString(72, height - 72, "INVOICE")
            c.drawString(72, height - 92, "From: Global Tech Solutions Inc.")
            c.drawString(72, height - 112, "To: Acme Corporation")
            c.save()
            print(f"Created a dummy invoice: '{sample_pdf_path}'")
        except ImportError:
            print(
                "Please install reportlab to create a sample PDF: pip install reportlab"
            )
            exit()

    processor = InvoiceProcessor(pdf_path=sample_pdf_path)
    processor.run()
