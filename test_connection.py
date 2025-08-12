import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi

# Disable proxy settings
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

# Load environment variables
load_dotenv()

# Print API key status (first 4 chars only)
api_key = os.getenv("DASHSCOPE_API_KEY")
if api_key:
    print(f"API key found: {api_key[:4]}...")
else:
    print("No API key found!")

# Try to initialize the model
try:
    llm = ChatTongyi(
        model_name="qwen-max",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    print("Successfully initialized ChatTongyi model")
except Exception as e:
    print(f"Error initializing model: {str(e)}")