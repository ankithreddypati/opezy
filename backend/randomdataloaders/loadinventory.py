from dotenv import load_dotenv
import os
import pymongo
from pymongo import MongoClient
from models import Inventory
import certifi
from datetime import datetime


# Load environment variables
load_dotenv(".env")
DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")

# Connect to MongoDB
db_client = MongoClient(DB_CONNECTION_STRING, tlsCAFile=certifi.where(),uuidRepresentation='standard')
db = db_client.opezy_works

# Define inventory entries
inventory_entries = [
    Inventory(
        InventoryName="zucchini",
        itemCost=2.5,
        itemDescription="Fresh zucchini",
        #images to emulate inventory images taken at regular intervals
        imagePath="images/zucc.jpeg"
    ),
    Inventory(
        InventoryName="guacamole",
        itemCost=1.5,
        itemDescription="Fresh guacamole",
        imagePath="images/guac1.jpeg"
    ),
    Inventory(
        InventoryName="Bell Peppers",
        itemCost=3.0,
        itemDescription="Crispy Red bell peppers in red color",
        imagePath="images/bellpepperrot2.png"
    ),
    Inventory(
        InventoryName="Marinara Sauce",
        itemCost=3.0,
        itemDescription="Red sauce",
        expiryDate=datetime(2024, 6, 30)
    ),
    Inventory(
        InventoryName="Salsa Verde ",
        itemCost=3.0,
        itemDescription="Salsa sauce",
        expiryDate=datetime(2024, 6, 28)
    ),

]

for item in inventory_entries:
    db.inventory.insert_one(item.model_dump(by_alias=True))

print("Inventory entries have been loaded into the database.")

