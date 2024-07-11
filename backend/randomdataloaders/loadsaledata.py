import os
import pymongo
import certifi
import random
from datetime import datetime
from faker import Faker
from dotenv import load_dotenv
from pymongo.errors import ServerSelectionTimeoutError

from models import Product, Ingredient, Sale, SaleDetail, Customer, Inventory, Feedback

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

# Define category-specific messages
category_messages = {
    'Burritos': ['Please cut into smaller pieces.'],
    'Tacos': ['Make it more spicy.', 'No garlic.'],
    'Drinks': ['Is there any other milk like almond milk, oat milk?','can you get other milk like oatmilk'],
    'Sweets': ['Less sugar, please.', 'No nuts.']  
}

# Retrieve customer IDs
customer_ids = [str(customer['_id']) for customer in db.customers.find()]

# Retrieve product details including their categories
product_details = list(db.products.find({}, {'_id': 1, 'base_price': 1, 'category': 1, 'available_add_ons': 1}))



fake = Faker()

def generate_random_sales(num_sales, customer_ids, product_details, category_messages):
    sales = []
    for _ in range(num_sales):
        customer_id = random.choice(customer_ids)
        sale_date = fake.date_time_between(start_date="-1y", end_date="now")
        
        num_products = random.randint(1, 3)
        sale_details = []
        total_amount = 0
        
        for _ in range(num_products):
            product = random.choice(product_details)
            quantity = random.randint(1, 10)
            purchased_price = product['base_price']
            product_category = product['category']
            
            # Pick a random custom message from the category-specific list
            special_instruction = random.choice(category_messages.get(product_category, [fake.sentence()]))
            
            # Calculate the total product price
            total_product_price = purchased_price * quantity
            total_amount += total_product_price
            
            # Append sale detail
            sale_detail = {
                "product_id": str(product['_id']),  
                "purchased_price": purchased_price,
                "quantity": quantity,
                "special_instructions": special_instruction
            }
            sale_details.append(sale_detail)
        
        # Create and append the complete sale entry
        sale = {
            "customer_id": customer_id,
            "date": sale_date,
            "sale_details": sale_details,
            "total_amount": total_amount
        }
        sales.append(sale)
    
    return sales

# Generating and inserting sales
random_sales = generate_random_sales(7000, customer_ids, product_details, category_messages)
db.sales.insert_many(random_sales)

print("Random sales with category-specific messages inserted successfully!")
