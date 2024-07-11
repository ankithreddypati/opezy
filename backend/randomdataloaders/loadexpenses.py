import os
import pymongo
import certifi
import random
from datetime import datetime, timedelta
from faker import Faker
from dotenv import load_dotenv
from pymongo.errors import ServerSelectionTimeoutError
from uuid import uuid4

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

fake = Faker()

expense_categories = {
    "Food Supplies": "Bulk grains and spices purchase",
    "Utilities": "Electricity bill payment",
    "Labor": "Monthly salary for kitchen staff",
    "Rent": "Monthly rental payment for restaurant space",
    "Maintenance": "Kitchen equipment repair",
    "Insurance": "Property insurance premium",
    "Miscellaneous": "Office supplies purchase",
    "Technology": "Software subscription fee",
    "Transport": "Fuel charges for delivery vehicles"
}

def generate_structured_monthly_expenses():
    expenses = []
    start_date = datetime.now() - timedelta(days=365)  

    for i in range(12):
        month_start = start_date + timedelta(days=30 * i)
        for category, description in expense_categories.items():
            expense_date = fake.date_time_between(start_date=month_start, end_date=month_start + timedelta(days=30))
            expense_amount = round(random.uniform(50, 1000), 2)  
            if i == 6 and category == "Labor":  
                expense_amount = round(random.uniform(5000, 10000), 2)

            expense = {
                "_id": str(uuid4()),
                "expenseDate": expense_date,
                "expenseCategory": category,
                "expenseAmount": expense_amount,
                "expenseDescription": description
            }
            expenses.append(expense)

    return expenses

db.expenses.delete_many({})
structured_expenses = generate_structured_monthly_expenses()
db.expenses.insert_many(structured_expenses)

print("Structured monthly expenses with an introduced anomaly in Labor inserted successfully!")
