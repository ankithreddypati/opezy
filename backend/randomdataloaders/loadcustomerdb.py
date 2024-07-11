import os
import pymongo
import certifi
import random
from datetime import datetime, timedelta
from uuid import uuid4
from models import Product, Ingredient, Sale, Customer, Inventory, Feedback  
from dotenv import load_dotenv
from pymongo.errors import ServerSelectionTimeoutError
from faker import Faker


# Environment setup and database connection
load_dotenv()
CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")


try:
    client = pymongo.MongoClient(
        CONNECTION_STRING,
        tlsCAFile=certifi.where(),
        tls=True,
        tlsAllowInvalidCertificates=True  
    )
    db = client['opezy_works']
    print("Connected to MongoDB.")
except ServerSelectionTimeoutError as err:
    print(f"Server selection timeout error: {err}")
    
    
customers_collection = db['customers']


# Generate customer data
def generate_customers(num_customers):
    fake = Faker()
    customers = []
    for _ in range(num_customers):
        customer = Customer(
            customerName=fake.name(),
            customerEmail="sreemukhi2502@gmail.com",
            customerPhone=fake.phone_number()
        )
        customers.append(customer)
    return customers

# Insert customers into MongoDB
customers = generate_customers(50)
for customer in customers:
    customers_collection.insert_one(customer.model_dump())

print("Customers inserted successfully!")