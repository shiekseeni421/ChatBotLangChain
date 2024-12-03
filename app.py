from flask import Flask, request, jsonify
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# Initialize Flask app
app = Flask(__name__)

# Load Google Gemini API key
genai.configure(api_key="")

# Initialize Chat model with the 'model' field
chat_model = ChatGoogleGenerativeAI(api_key="", model="gemini-1.5-flash")

# Prompt template for question answering
prompt = PromptTemplate(input_variables=["question"], template="Answer the following question:\n\n{question}\n\nAnswer:")

# Initialize LLMChain with the prompt and model
llm_chain = LLMChain(llm=chat_model, prompt=prompt)


@app.route('/query', methods=['POST'])
def query():
    """
    Directly query the Google Gemini LLM for a response based on user input.
    Expects JSON payload with a 'question' key.
    """
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        # Generate the answer using LangChain and Google Gemini LLM
        answer = llm_chain.run({"question": question})
        return jsonify({"answer": answer}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
