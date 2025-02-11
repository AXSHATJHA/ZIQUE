import os
import pandas as pd
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from typing import List
from typing_extensions import TypedDict
from groq import Groq
from langgraph.graph import END, StateGraph, START # type: ignore
from typing import Dict
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
from langgraph.checkpoint.memory import MemorySaver # type: ignore
from google import generativeai as genai
from dotenv import load_dotenv
import openai # type: ignore
from openai import OpenAI # type: ignore

df = pd.read_csv('Gigi Food Menu Database.csv')
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Create the model
generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}


class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "general_chatbot"] = Field(
        ...,
        description="""
      ZICO, a food recommendation chatbot.
      """
    )


llm = ChatGroq(model="llama3-8b-8192", api_key="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")
structured_llm_router = llm.with_structured_output(RouteQuery)
system = """
      You are an expert dish recommendation router. Analyze user questions and route them to the most appropriate resource:
        1. **Food Menu Database (Vectorstore):**
            - Questions about specific dishes, ingredients, or dietary options
            - Dish Categories:
                * Cold (Salads, Ceviche)
                * Hot (Soups, Grilled Dishes)
                * Desserts (Cakes, Ice Creams)
                * Beverages (Juices, Smoothies)
            - Filters available:
                * Diet Type (Vegetarian, Vegan, Keto, Gluten-Free)
                * Cuisine Type (European, Japanese, Fusion, etc.)
                * Meal Type (Breakfast, Lunch, Dinner)
            - Examples:
                - "Show me vegetarian salads"
                - "Dishes with avocado"
                - "Gluten-free desserts"
                - "Suggest a healthy sushi"
                - "more", "more dishes"
                - "Suggest healthy options"
                - "What dishes do you offer?"
        2. **general_chatbot (Direct Response):**
            - Greetings, help with menu navigation, or general inquiries
            - Examples:
                - "Hi"
                - "Hello"
                - "Namaste"
                - "Hi, my name is (name)"
                - "What is my name?"
                - "How do I use the chatbot?"
                - "Which meals are dairy-free?"
        Always respond in JSON format:
        {{
          "datasource": "vectorstore|general_chatbot",
          "reasoning": "Brief explanation of routing decision"
        }}
        - Special Considerations:
        - Prioritize vocational dishes (e.g., Cooking techniques, Regional Cuisines)
        - Suggest meal options based on dietary restrictions (e.g., Diabetes-friendly, High-protein)
        - Handle multi-language queries if dish descriptions are available in Hindi/English
        - Consider age restrictions (e.g., Alcoholic beverages for 18+)
        }}
      """

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
question_router = route_prompt | structured_llm_router

api_wrapper = WikipediaAPIWrapper(top_k=3, doc_content_chars_max=1000)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)


class GraphState(TypedDict):
    question: str
    generation: str
    docs: List[str]
    chat_history: List[Dict[str, str]]  # Add chat history


def route_question(state):
    print("---Route Question----")
    question = state["question"]

    # Check if the question is a general query (e.g., greetings, casual questions)

    # Otherwise, route to database or Wikipedia search
    source = question_router.invoke({"question": question})
    if source.datasource == "wiki_search":
        print("---ROUTE QUESTION To Wiki Search---")
        return "wiki_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION To Vectorstore---")
        return "vectorstore"

    elif source.datasource == "general_chatbot":
        print("---ROUTE QUESTION To General Chatbot---")
        return "general_chatbot"


workflow = StateGraph(GraphState)

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "vectorstore": END,
        "general_chatbot": END
    },
)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

api = Flask(__name__)
CORS(api)

groq_client = Groq(api_key="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")


@api.route("/chat/", methods=["POST"])
def chat():
    try:
        user_input = request.get_json()
        question = user_input.get("question")
        thread_id = user_input.get("thread_id", f"thread_{random.randint(1000, 9999)}")

        if not question:
            return jsonify({"error": "Question is required."}), 400

        chat_history = user_input.get("chat_history", [])

        # Route the question
        source = question_router.invoke({"question": question})
        print(f"Routing decision: {source.datasource}")

        if source.datasource == "general_chatbot":
            print("---General Chatbot----")

            # Define the system prompt for the chatbot
            system_prompt = """
            Your name is Zico(always say it if greeted).
            You are a friendly and helpful female AI assistant designed for general conversation and greeting purposes. 
            Always respond in a warm, polite, and informative manner. If the user asks a question, provide a clear and concise answer. 
            If the user engages in casual conversation, respond appropriately to keep the conversation flowing naturally.
            """

            # Prepare the conversation history (last 5 messages)
            last_5_messages = chat_history[-5:] if chat_history else []

            # Construct the messages for the GPT API call
            messages = [
                           {"role": "system", "content": system_prompt}
                       ] + last_5_messages + [
                           {"role": "user", "content": question}
                       ]

            # Make the API call to GPT
            response = openai_client.chat.completions.create(
                model="gpt-4",  # Use GPT-4 for better conversational abilities
                messages=messages,
                temperature=0.4,  # Moderate temperature for balanced responses
                max_tokens=400  # Limit response length
            )

            # Extract the chatterbot's response
            if response.choices and len(response.choices) > 0:
                chatbot_response = response.choices[0].message.content
            else:
                chatbot_response = "I'm sorry, I couldn't generate a response. Please try again."

            # Update the chat history
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": chatbot_response})
            chat_history = chat_history[-10:]  # Keep only the last 10 messages

            # Return the response and updated chat history
            return jsonify({"response": chatbot_response, "chat_history": chat_history})

        elif source.datasource == "vectorstore":
            print("---Handling Vectorstore Query---")

            # Load dataset
            df = pd.read_csv("Gigi Food Menu Database.csv")  # Assuming CSV file

            # Convert the dataset into a readable string format
            dataset_str = "Dish Category | Dish Name | Diet Type | Allergens | Cuisine Type | Staple Ingredients | Meal Type | Flavor Profile\n"
            dataset_str += "------------------------------------------------------------------------------------------------------------------------\n"
            for _, row in df.iterrows():
                dataset_str += (
                    f"{row['Dish Category']} | {row['Dish Name']} | {row['Diet Type']} | {row['Allergens']} | "
                    f"{row['Cuisine Type']} | {row['Staple Ingredients']} | {row['Meal Type']} | "
                    f"{row['Flavor Profile']}\n"
                )

            # Initialize user preferences
            user_prefs = {
                'diet': set(),
                'allergens': set(),
                'flavors': set(),
                'meal_type': set()
            }

            # Extract user palate from chat history
            for msg in chat_history:
                if msg["role"] == "user":
                    content = msg["content"].lower()

                    # Check if the message contains "USER PALATE"
                    if "user palate" in content:
                        # Extract the palate information
                        palate_info = content.replace("user palate:", "").strip()

                        # Split into individual preferences
                        for pref in palate_info.split(","):
                            pref = pref.strip()
                            if "diet type:" in pref:
                                diet = pref.replace("diet type:", "").strip()
                                user_prefs['diet'].add(diet)
                            elif "allergy:" in pref:
                                allergen = pref.replace("allergy:", "").strip()
                                user_prefs['allergens'].add(allergen)
                            elif "flavor:" in pref:
                                flavor = pref.replace("flavor:", "").strip()
                                user_prefs['flavors'].add(flavor)
                            elif "meal type:" in pref:
                                meal_type = pref.replace("meal type:", "").strip()
                                user_prefs['meal_type'].add(meal_type)

            # Construct the prompt with the entire dataset and user preferences
            prompt = f"""ROLE: You are Zico, a culinary expert at Gigi.
                TASK: Recommend EXACTLY 2 dishes based on the user's palate, dietary preferences, allergens, and the current question.

                DATASET:
                {dataset_str}

                USER PALATE:
                - Diet: {', '.join(user_prefs['diet']) if user_prefs['diet'] else 'No specific diet'}
                - Allergens to avoid: {', '.join(user_prefs['allergens']) if user_prefs['allergens'] else 'None'}
                - Preferred flavors: {', '.join(user_prefs['flavors']) if user_prefs['flavors'] else 'Not specified'}
                - Meal type: {', '.join(user_prefs['meal_type']) if user_prefs['meal_type'] else 'Not specified'}

                CURRENT QUESTION:
                "{question}"

                INSTRUCTIONS:
                - If asked for more or more dishes or something like that SUGGEST ACCORDING TO THE LAST PROMPT THAT WAS 
                    GIVEN. SUGGEST DISHES THAT WERE SUGGESTED EARLIER. Here is the chat history{chat_history[-1:]}  
                - Prioritize dishes that match the user's dietary preferences and avoid allergens.
                - If the user has specifically asked for a dish category (e.g., pizza, pasta, sushi, ice cream), recommend only dishes from that category.
                - If the user has specified preferred flavors or meal types, prioritize dishes that match those.
                - Ensure the recommendations are relevant to the user's query.
                - Do not recommend dishes that contain allergens the user wants to avoid.
                RECOMMEND ONLY 1 DISH
                FORMAT STRICTLY AS:
                 **Dish Name**
                   - Category: [Dish Category]
                   - Diet Type: [Diet Type]
                   - Key flavors: [Flavor Profile]
                   - Ingredients: [3-5 staple ingredients]
                   - Why: [15-word explanation of why this dish matches the user's preferences and query]

                NEVER mention allergens. Avoid markdown."""

            # Make API call to OpenAI
            openai.api_key = os.environ["OPENAI_API_KEY"]
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use GPT-4 for better understanding of the dataset
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1  # Lower temperature for more focused recommendations
            )

            initial_message = response.choices[0].message.content

            refined_response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system",
                     "content": "You are ZICO, a female chatbot that specializes in recommending a dish to a user in a very beautiful and effective manner according to the question and satisfies the user. You should answer in around 20 words with a greeting and telling why the user should have the dish. Answer in a table kind of format so the dish is highlighted."},
                    {"role": "user",
                     "content": f"Here is user's question : {question}. and here is the recommended dish to the user: {initial_message}. Now it is your job to give the user its recommendation like a female chatbot ZICO."}
                ],
                temperature=0.1
            )

            # Extract the refined response content
            final_message = refined_response.choices[0].message.content

            # Update chat history
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": final_message})
            chat_history = chat_history[-10:]

            return jsonify({"response": final_message, "chat_history": chat_history})

        else:
            return jsonify({"error": "Invalid datasource."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == "__main__":
    api.run(host="0.0.0.0", port=8000)
