import os
import json
from typing import List
from datetime import datetime, timedelta
from bson import ObjectId
from collections import defaultdict
import pandas as pd
import tweepy
import pymongo
import base64
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch
from langchain.schema.document import Document
from langchain.agents import Tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain_core.messages import SystemMessage
from langchain.tools import StructuredTool
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import google.auth.exceptions

load_dotenv(".env")
DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
AOAI_ENDPOINT = os.environ.get("AOAI_ENDPOINT")
AOAI_KEY = os.environ.get("AOAI_KEY")
consumer_key = os.getenv('TWITTER_API_KEY')
consumer_secret = os.getenv('TWITTER_API_SECRET')
bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
access_token = os.getenv('TWITTER_ACCESS_TOKEN')
access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
AOAI_API_VERSION = "2023-09-01-preview"
COMPLETIONS_DEPLOYMENT = "gpt-35-turbo"
EMBEDDINGS_DEPLOYMENT = "text-embedding-ada-002"
db_client = pymongo.MongoClient(DB_CONNECTION_STRING, tlsAllowInvalidCertificates=True)
db = db_client.opezy_works



def get_gmail_credentials(token_file, scopes, client_secrets_file):
    creds = None
    try:
        if os.path.exists(token_file):
            creds = Credentials.from_authorized_user_file(token_file, scopes)
        
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(token_file, 'w') as token:
                token.write(creds.to_json())
    except google.auth.exceptions.RefreshError as e:
        print(f"Failed to refresh access token: {e}")
        creds = None 
    except Exception as e:
        print(f"An error occurred during Gmail authentication: {e}")
        creds = None

    if not creds or not creds.valid:
        try:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
            creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(token_file, 'w') as token:
                token.write(creds.to_json())
        except Exception as e:
            print(f"An error occurred during Gmail authentication: {e}")
            creds = None
    
    return creds

# Initialize Gmail credentials and toolkit
try:
    credentials = get_gmail_credentials(
        token_file="./opezy/token.json",
        scopes=["https://mail.google.com/"],
        client_secrets_file="./opezy/credentials.json"
    )
    if credentials:
        api_resource = build_resource_service(credentials=credentials)
        gmail_toolkit = GmailToolkit(api_resource=api_resource)
    else:
        gmail_toolkit = None
except Exception as e:
    print(f"Failed to initialize Gmail toolkit: {e}")
    gmail_toolkit = None


# V1 Twitter API Authentication
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# V2 Twitter API Authentication
client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=consumer_key,
    consumer_secret=consumer_secret,
    access_token=access_token,
    access_token_secret=access_token_secret,
    wait_on_rate_limit=True
)

api_resource = build_resource_service(credentials=credentials)
gmail_toolkit = GmailToolkit(api_resource=api_resource)

class OpezyAIAgent:
    def __init__(self, session_id: str):
        self.embedding_model = AzureOpenAIEmbeddings(
            openai_api_version=AOAI_API_VERSION,
            azure_endpoint=AOAI_ENDPOINT,
            openai_api_key=AOAI_KEY,
            azure_deployment=EMBEDDINGS_DEPLOYMENT,
            chunk_size=10
        )
        self.gmail_toolkit = gmail_toolkit

        system_message = SystemMessage(
            content="""
                You are an AI assistant named 'Opezy' designed to help the owner with queries about her Mexican restaurant.
                
                The owner name is Garcia.
                
                You are designed to answer questions about the products, sales, customers, expenses, Inventory, marketing, communication and customer feedback.   

                If you don't know the answer to a question, respond with "I don't know."
                
                If a question is not related to restaurant or business in general, respond with "I only answer questions about restaurant business."
            """
        )

        self.agent_executor = create_conversational_retrieval_agent(
            AzureChatOpenAI(
                temperature=0,
                openai_api_version=AOAI_API_VERSION,
                azure_endpoint=AOAI_ENDPOINT,
                openai_api_key=AOAI_KEY,
                azure_deployment=COMPLETIONS_DEPLOYMENT
            ),
            self.__create_agent_tools(),
            system_message=system_message,
            memory_key=session_id,
            verbose=True
        )

    def run(self, prompt: str) -> str:
        try:
            result = self.agent_executor({"input": prompt})
            print(f"Raw Result: {result}")  
            output = result["output"]
            print(f"Processed Output: {output}") 
            return output
        except Exception as e:
            print(f"Exception occurred: {e}")
            return "An error occurred while processing your request."

    def __create_opezy_works_vector_store_retriever(
        self,
        collection_name: str,
        top_k: int = 3
    ):
        vector_store = AzureCosmosDBVectorSearch.from_connection_string(
            connection_string=DB_CONNECTION_STRING,
            namespace=f"opezy_works.{collection_name}",
            embedding=self.embedding_model,
            index_name="VectorSearchIndex",
            embedding_key="contentVector",
            text_key="_id"
        )
        return vector_store.as_retriever(search_kwargs={"k": top_k})
    
    def __create_agent_tools(self) -> List[Tool]:
        feedback_retriever = self.__create_opezy_works_vector_store_retriever("feedbacks")
        feedback_retriever_chain = feedback_retriever | format_docs

        most_sold_product_tool = StructuredTool.from_function(
            func=get_most_sold_product,
            name="most_sold_product_tool",
            description="Retrieves the most sold product."
        )
        
        least_sold_product_tool = StructuredTool.from_function(
            func=get_least_sold_product,
            name="least_sold_product_tool",
            description="Retrieves the most sold product."
        )

        declining_sales_tool = StructuredTool.from_function(
            func=get_declining_products,
            name="declining_sales_report",
            description="Retrieves products with declining sales in dollars on a monthly basis."
        )

        feedback_tool = StructuredTool.from_function(
            func=get_feedback_comments_by_keyword,
            name="vector_search_feedback",
            description="Searches Opezy Works product feedback based on a keyword."
        )
        
        demand_forecasting_tool = StructuredTool.from_function(
            func=predict_demand_for_tomorrow,
            name="demand_forecast",
            description="Predicts the demand for given product for tomorrow"
        )
        
        quality_check_tool = StructuredTool.from_function(
            func=check_product_freshness,
            name="quality_check",
            description="Checks the freshness of a inventory item using images"
        )
        
        expense_anomaly_tool = StructuredTool.from_function(
            func=detect_expense_anomalies,
            name="expense_anomaly_detection",
            description="Detects anomalies in expenses"
        )
        
        
        increasing_expenses_tool = StructuredTool.from_function(
            func=get_increasing_expenses,
            name="increasing_expenses_tool",
            description="Identifies expense categories with increasing amounts on a monthly basis."
        )
        
        tweet_tool = StructuredTool.from_function(
            func=tweet_marketing_post,
            name="tweet_marketing_post",
            description="Tweets a marketing post."
        )
        
        # gmail_tools = self.gmail_toolkit.get_tools()
        
        marketing_email_tool = StructuredTool.from_function(
            func=send_marketing_email,
            name="send_marketing_email",
            description="Sends marketing emails to all customers."
        )
        
        inventory_expiry_tool = StructuredTool.from_function(
            func=check_inventory_expiring_soon,
            name="check_inventory_expiring_soon",
            description="Check expiry date of the invetory."
        )
        
        
    
        tools = [
            most_sold_product_tool,
            least_sold_product_tool,
            declining_sales_tool,
            feedback_tool,
            demand_forecasting_tool,
            quality_check_tool,
            expense_anomaly_tool,
            inventory_expiry_tool,
            increasing_expenses_tool,
            tweet_tool,
            marketing_email_tool,
            
          
        ]
        
        if self.gmail_toolkit:
            tools.extend(self.gmail_toolkit.get_tools())
            
        return tools

def tweet_marketing_post(text: str, image_path: str = None) -> str:
    """
    Tweets a marketing post using Twitter API v2, optionally including an image.
    :param text: The text to tweet.
    :param image_path: The path to the image to upload and include in the tweet (optional).
    :return: A JSON string with the result of the tweet.
    """
    try:
        if image_path:
            media_id = api.media_upload(filename=image_path).media_id_string
            response = client.create_tweet(text=text, media_ids=[media_id])
        else:
            response = client.create_tweet(text=text)
        return f"Tweet created successfully: {response.data}"
    except Exception as e:
        return f"Error occurred: {e}"



def get_customer_emails() -> List[str]:
    """
    Fetches all customer emails from the MongoDB collection.
    :return: A list of email addresses.
    """
    customers = db.customers.find({}, {'email': 1})
    email_list = [customer['email'] for customer in customers if 'email' in customer]
    return email_list

def send_marketing_email(subject: str, body: str) -> str:
    """
    Sends a marketing email to all customers.
    :param subject: The subject of the email.
    :param body: The body content of the email.
    :return: Status message.
    """
    emails = get_customer_emails()
    if not emails:
        return "No customer emails found."

    # Get the email sending tool from GmailToolkit
    gmail_tools = gmail_toolkit.get_tools()
    send_email_tool = None
    for tool in gmail_tools:
        if tool.name == 'send_gmail_message':
            send_email_tool = tool
            break
    
    if send_email_tool is None:
        return "Email sending tool not found."

    success_count = 0
    failure_count = 0

    for email in emails:
        try:
            send_email_tool.run({
                'to': email,
                'subject': subject,
                'body': body
            })
            print(f"Email sent to: {email}")
            success_count += 1
        except Exception as e:
            print(f"Failed to send email to: {email}, error: {e}")
            failure_count += 1
    
    return f"Marketing emails sent: {success_count} succeeded, {failure_count} failed."


def check_inventory_expiring_soon(days: int = 7) -> str:
    """
    Checks for inventory items that are expiring within the specified number of days.
    :param days: Number of days from today to check for expiring inventory.
    :return: A JSON string with the expiring inventory items.
    """
    try:
        # Calculate the threshold date
        current_date = datetime.now()
        threshold_date = current_date + timedelta(days=days)
        
        # Query the inventory collection for items expiring soon
        expiring_items = list(db.inventory.find({
            "expirydate": {
                "$lte": threshold_date
            }
        }, {
            "name": 1, "expirydate": 1, "_id": 0  # Project only the necessary fields
        }))
        
        if not expiring_items:
            return json.dumps({"message": f"No inventory items expiring within the next {days} days."})

        return json.dumps(expiring_items, default=str)
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return json.dumps({"error": str(e)})
    
def get_increasing_expenses(start_date: str, end_date: str) -> str:
    """
    Retrieves expense categories with generally increasing trends over the specified period.
    """
    start_date = datetime.fromisoformat(start_date)
    end_date = datetime.fromisoformat(end_date)

    pipeline = [
        {
            '$match': {
                'expenseDate': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
        },
        {
            '$group': {
                '_id': {
                    'category': '$expenseCategory',
                    'month': { '$dateToString': { 'format': '%Y-%m', 'date': '$expenseDate' } }
                },
                'totalExpense': { '$sum': '$expenseAmount' }
            }
        },
        {
            '$sort': {
                '_id.category': 1,
                '_id.month': 1
            }
        }
    ]

    try:
        result = list(db.expenses.aggregate(pipeline))
        expense_data = defaultdict(list)

        for record in result:
            category = record['_id']['category']
            month = record['_id']['month']
            total_expense = record['totalExpense']
            expense_data[category].append(total_expense)

        increasing_expenses = []

        # Analyze overall trend rather than month-to-month
        for category, expenses in expense_data.items():
            if len(expenses) > 1 and expenses[0] < expenses[-1]:  # Check if the last month's expense is greater than the first
                increasing_expenses.append({
                    'category': category,
                    'first_month_expense': expenses[0],
                    'last_month_expense': expenses[-1]
                })

        if not increasing_expenses:
            return json.dumps({"message": "No categories with increasing trends found."})

        return json.dumps(increasing_expenses, default=str)

    except Exception as e:
        print("Error occurred:", e)
        return json.dumps({"error": str(e)})




def detect_expense_anomalies(start_date: str, end_date: str) -> str:
    """
    Detects anomalies in expenses within a specified date range using both expense amounts and content vectors.
    :param start_date: Start date in ISO format (YYYY-MM-DD).
    :param end_date: End date in ISO format (YYYY-MM-DD).
    :return: A JSON string with detected anomalies.
    """
    start_date = datetime.fromisoformat(start_date)
    end_date = datetime.fromisoformat(end_date)

    try:
        expenses = list(db.expenses.find({
            "expenseDate": {"$gte": start_date, "$lte": end_date}
        }))

        if not expenses:
            return json.dumps({"message": "No expense data found in the specified date range."})

        # Prepare DataFrame
        df = pd.DataFrame(expenses)
        df['expenseAmount'] = df['expenseAmount'].astype(float)
        
        # Assuming 'contentVector' is stored as a list of floats
        vectors = np.array(df['contentVector'].tolist())

        # Standardizing the vectors (important for anomaly detection algorithms)
        scaler = StandardScaler()
        scaled_vectors = scaler.fit_transform(vectors)

        # Dimensionality Reduction (optional based on performance and accuracy needs)
        pca = PCA(n_components=10)  
        reduced_vectors = pca.fit_transform(scaled_vectors)

        # Combine scaled amount and reduced vectors
        amounts_scaled = scaler.fit_transform(df[['expenseAmount']])
        combined_features = np.hstack((amounts_scaled, reduced_vectors))

        # Anomaly Detection
        model = IsolationForest(n_estimators=100, contamination=0.05)
        predictions = model.fit_predict(combined_features)
        df['anomaly'] = predictions

        anomalies = df[df['anomaly'] == -1]

        if anomalies.empty:
            return json.dumps({"message": "No anomalies detected in expenses."})

        # Returning essential fields, excluding vectors
        anomalies = anomalies[["_id", "expenseDate", "expenseCategory", "expenseAmount", "expenseDescription"]]
        return anomalies.to_json(orient="records")
    except Exception as e:
        print(f"Error occurred: {e}")
        return json.dumps({"error": str(e)})   

def predict_demand_for_tomorrow(product_name: str) -> str:
    """
    Predicts the demand for a given product for tomorrow.
    :param product_name: The name of the product.
    :return: A JSON string with the predicted demand.
    """
    try:
        # Fetch the product_id based on product_name
        product = db.products.find_one({'name': product_name}, {'_id': 1})
        if not product:
            return json.dumps({"message": "Product not found."})
        
        product_id = product['_id']
        print(f"Product ID for '{product_name}': {product_id}")

        # Fetch historical sales data for the product
        sales_data = list(db.sales.aggregate([
            {'$unwind': '$sale_details'},
            {'$match': {'sale_details.product_id': str(product_id)}},  # Ensure product_id is string
            {'$group': {
                '_id': {'date': '$date'},
                'total_sales': {'$sum': '$sale_details.quantity'}
            }},
            {'$sort': {'_id.date': 1}}
        ]))

        if not sales_data:
            print(f"No sales data found for product_id: {product_id}")
            return json.dumps({"message": "No sales data found for the product."})

        print(f"Sales Data: {sales_data}")

        # Prepare the data for modeling
        df = pd.DataFrame(sales_data)
        df['date'] = pd.to_datetime(df['_id'].apply(lambda x: x['date']))
        df.set_index('date', inplace=True)
        df['day_of_week'] = df.index.dayofweek

        print(f"DataFrame: {df}")

        # Features and target
        X = df[['day_of_week']]
        y = df['total_sales']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Validate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

        # Predict demand for tomorrow
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow_day_of_week = pd.DataFrame({'day_of_week': [tomorrow.weekday()]})
        predicted_demand = model.predict(tomorrow_day_of_week)[0]

        print(f"Predicted Demand for {product_name}: {predicted_demand}")

        return json.dumps({"product_name": product_name, "predicted_demand": int(predicted_demand)}, default=str)

    except Exception as e:
        print("Error occurred:", e)
        return json.dumps({"error": str(e)})



def get_feedback_comments_by_keyword(keywords: List[str]) -> str:
    """
    Retrieves feedback comments that contain any of the specified keywords.
    :param keywords: A list of keywords to search for in the feedback comments.
    :return: A JSON string with the matching feedback comments.
    """
    regex_patterns = [{"feedbackComment": {"$regex": keyword, "$options": "i"}} for keyword in keywords]
    
    # Use the $or operator to match any of the patterns
    feedbacks = db.feedbacks.find({"$or": regex_patterns})
    
    result = []
    for feedback in feedbacks:
        if "contentVector" in feedback:
            del feedback["contentVector"]
        result.append(feedback)
    
    return json.dumps(result, default=str)

def format_docs(docs: List[Document]) -> str:
    str_docs = []
    for doc in docs:
        if isinstance(doc.page_content, str):  
            doc_dict = {"_id": doc.page_content}
            doc_dict.update(doc.metadata)
            if "contentVector" in doc_dict:
                del doc_dict["contentVector"]
            str_docs.append(json.dumps(doc_dict, default=str))
        else:
            doc_dict = {"_id": str(doc.page_content)}
            doc_dict.update(doc.metadata)
            if "contentVector" in doc_dict:
                del doc_dict["contentVector"]
            str_docs.append(json.dumps(doc_dict, default=str))
    return "\n\n".join(str_docs)


def get_most_sold_product(start_date: str, end_date: str) -> str:
    """
    Retrieves the most sold product in the given date range based on total sales (quantity x price).
    :param start_date: The start date of the interval in ISO format.
    :param end_date: The end date of the interval in ISO format.
    :return: A JSON string with the results.
    """
    start_date = datetime.fromisoformat(start_date)
    end_date = datetime.fromisoformat(end_date)
    
    pipeline = [
        {
            '$unwind': '$sale_details'
        },
        {
            '$match': {
                'date': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
        },
        {
            '$group': {
                '_id': '$sale_details.product_id',
                'totalSales': {
                    '$sum': {
                        '$multiply': [
                            '$sale_details.purchased_price',
                            '$sale_details.quantity'
                        ]
                    }
                }
            }
        },
        {
            '$sort': {
                'totalSales': -1
            }
        },
        {
            '$limit': 1
        }
    ]
    
    try:
        print("Executing MongoDB Aggregation Pipeline:")
        result = list(db.sales.aggregate(pipeline))
        print("Pipeline Result:", result)
        
        if not result:
            return json.dumps({"message": "No sales data found."})
        
        most_sold_product = result[0]
        product_id = most_sold_product['_id']
        
        product_details = db.products.find_one({'_id': ObjectId(product_id)}, {'name': 1, '_id': 0})
        
        if not product_details:
            return json.dumps({"message": "Product details not found."})
        
        most_sold_product['product_name'] = product_details['name']
        most_sold_product['total_sales_value'] = most_sold_product['totalSales']
        return json.dumps(most_sold_product, default=str)
    
    except Exception as e:
        print("Error occurred:", e)
        return json.dumps({"error": str(e)})
    
def get_least_sold_product(start_date: str, end_date: str) -> str:
    """
    Retrieves the least sold product in the given date range based on total sales (quantity x price).
    :param start_date: The start date of the interval in ISO format.
    :param end_date: The end date of the interval in ISO format.
    :return: A JSON string with the results.
    """
    start_date = datetime.fromisoformat(start_date)
    end_date = datetime.fromisoformat(end_date)
    
    pipeline = [
        {
            '$unwind': '$sale_details'
        },
        {
            '$match': {
                'date': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
        },
        {
            '$group': {
                '_id': '$sale_details.product_id',
                'totalSales': {
                    '$sum': {
                        '$multiply': [
                            '$sale_details.purchased_price',
                            '$sale_details.quantity'
                        ]
                    }
                }
            }
        },
        {
            '$sort': {
                'totalSales': 1
            }
        },
        {
            '$limit': 1
        }
    ]
    
    try:
        print("Executing MongoDB Aggregation Pipeline:")
        result = list(db.sales.aggregate(pipeline))
        print("Pipeline Result:", result)
        
        if not result:
            return json.dumps({"message": "No sales data found."})
        
        most_sold_product = result[0]
        product_id = most_sold_product['_id']
        
        product_details = db.products.find_one({'_id': ObjectId(product_id)}, {'name': 1, '_id': 0})
        
        if not product_details:
            return json.dumps({"message": "Product details not found."})
        
        most_sold_product['product_name'] = product_details['name']
        most_sold_product['total_sales_value'] = most_sold_product['totalSales']
        return json.dumps(most_sold_product, default=str)
    
    except Exception as e:
        print("Error occurred:", e)
        return json.dumps({"error": str(e)})

def get_declining_products(start_date: str, end_date: str) -> str:
    """
    Retrieves products with declining sales on a monthly basis.
    :param start_date: The start date of the interval in ISO format.
    :param end_date: The end date of the interval in ISO format.
    :return: A JSON string with the results.
    """
    
    start_date = datetime.fromisoformat(start_date)
    end_date = datetime.fromisoformat(end_date)

    pipeline = [
        {
            '$unwind': '$sale_details'
        },
        {
            '$match': {
                'date': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
        },
        {
            '$group': {
                '_id': {
                    'product_id': '$sale_details.product_id',
                    'month': { '$dateToString': { 'format': '%Y-%m', 'date': '$date' } }
                },
                'totalSales': {
                    '$sum': {
                        '$multiply': [
                            '$sale_details.purchased_price',
                            '$sale_details.quantity'
                        ]
                    }
                }
            }
        },
        {
            '$sort': {
                '_id.product_id': 1,
                '_id.month': 1
            }
        }
    ]
    
    try:
        print("Executing MongoDB Aggregation Pipeline:")
        result = list(db.sales.aggregate(pipeline))
        print("Pipeline Result:", result)
        
        if not result:
            return json.dumps({"message": "No sales data found."})
        
        sales_data = defaultdict(list)
        for record in result:
            product_id = record['_id']['product_id']
            month = record['_id']['month']
            total_sales = record['totalSales']
            sales_data[product_id].append((month, total_sales))
        
        declining_products = []
        
        for product_id, monthly_sales in sales_data.items():
            sales_values = [sales for month, sales in monthly_sales]
            if all(earlier >= later for earlier, later in zip(sales_values, sales_values[1:])):
                product_details = db.products.find_one({'_id': ObjectId(product_id)}, {'name': 1, '_id': 0})
                if product_details:
                    formatted_sales = [(month, f"${sales:.2f}") for month, sales in monthly_sales]
                    declining_products.append({
                        'product_id': product_id,
                        'product_name': product_details['name'],
                        'monthly_sales': formatted_sales
                    })
        
        if not declining_products:
            return json.dumps({"message": "No products with declining sales found."})
        
        return json.dumps(declining_products, default=str)
    
    except Exception as e:
        print("Error occurred:", e)
        return json.dumps({"error": str(e)})
           

def fetch_image_path(inventory_name: str) -> str:
    inventory_item = db.inventory.find_one({'InventoryName': {'$regex': f'^{inventory_name}$', '$options': 'i'}}, {'imagePath': 1, '_id': 0})
    if not inventory_item or 'imagePath' not in inventory_item:
        raise ValueError(f"No image path found for inventory item: {inventory_name}")
    return inventory_item['imagePath']


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def check_product_freshness(inventory_name: str) -> str:
    try:
        image_path = fetch_image_path(inventory_name)

        base64_image = encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "api-key": AOAI_KEY
        }

        # Prepare payload
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Is this {inventory_name} fresh looking?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post(f"{AOAI_ENDPOINT}/openai/deployments/gpt-4-vision/chat/completions?api-version={AOAI_API_VERSION}", headers=headers, json=payload)

        return response.json()

    except Exception as e:
        print("Error occurred:", e)
        return json.dumps({"error": str(e)})