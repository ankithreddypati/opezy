import os
import pymongo
import certifi
import random
from datetime import datetime, timedelta
from uuid import uuid4
from models import Product, Ingredient, Sale, Customer, Inventory, Feedback  
from dotenv import load_dotenv
from pymongo.errors import ServerSelectionTimeoutError

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
    
    
from models import Product, Ingredient

categories = {
    "Tacos": "Authentic Mexican street-style tacos with various fillings.",
    "Burritos": "Hearty burritos wrapped in soft tortillas, customizable with add-ons.",
    "Drinks": "Refreshing beverages inspired by Mexican flavors.",
    "Sweets": "Traditional sweets"
}

mexican_products = []

taco_products = [
    Product(
        productName="Taco al Pastor",
        productCategory="Tacos",
        productPrice=4.0,
        productDescription="Marinated pork, pineapple, and onions in a soft corn tortilla.",
        availableAddOns=[
            Ingredient(name="Salsa Verde",extra_cost=1.0),
            Ingredient(name="Cilantro",extra_cost=1.0),
            Ingredient(name="Onions",extra_cost=1.0)
        ]
    ),
    Product(
        productName="Barbacao Taco",
        productCategory="Tacos",
        productPrice=4.0,
        productDescription="Tender barbacoa beef in a warm flour tortilla, garnished with chopped cilantro, white onions, and a dash of salsa verde.",
        availableAddOns=[
            Ingredient(name="Sour Cream",extra_cost=1.0),
            Ingredient(name="Cheddar Cheese",extra_cost=1.0),
            Ingredient(name="Avocado Slices",extra_cost=1.0)
        ]
    ),
    Product(
        productName="Pollo Asado Taco",
        productCategory="Tacos",
        productPrice=5.0,
        productDescription="Grilled marinated chicken topped with lettuce, pico de gallo, and crema, wrapped in a handmade corn tortilla",
        availableAddOns=[
            Ingredient(name="Queso Fresco",extra_cost=1.0),
            Ingredient(name="Sliced Radishes",extra_cost=1.0),
            Ingredient(name="Salsa Roja",extra_cost=1.0)
        ]
    ),
     Product(
        productName="Fish Taco",
        productCategory="Tacos",
        productPrice=4.0,
        productDescription="Crispy beer-battered fish with cabbage slaw and chipotle mayo in a corn tortilla.",
        availableAddOns=[
            Ingredient(name="Mango Salsa",extra_cost=1.0),
            Ingredient(name="Avocado Cream",extra_cost=1.0),
            Ingredient(name="Extra Fish",extra_cost=1.0)
        ]
    )
]

burrito_products = [
    Product(
        productName="Chicken Burrito",
        productCategory="Burritos",
        productPrice=10.0,
        productDescription="Grilled chicken, rice, beans, and veggies wrapped in a flour tortilla.",
        availableAddOns=[
            Ingredient(name="Guacamole",extra_cost=1.0),
            Ingredient(name="Pico de Gallo",extra_cost=1.0),
            Ingredient(name="Lettuce",extra_cost=1.0)
        ]
    ),
    Product(
        productName="Breakfast Burrito",
        productCategory="Burritos",
        productPrice=10.0,
        productDescription="Scrambled eggs, chorizo, diced potatoes, and cheddar cheese in a flour tortilla, served all day",
        availableAddOns=[
            Ingredient(name="Guacamole",extra_cost=1.0),
            Ingredient(name="Salsa Roja",extra_cost=1.0),
            Ingredient(name="Extra Chorizo",extra_cost=1.0)
        ]
    ),
    Product(
        productName="Shrimp Burrito",
        productCategory="Burritos",
        productPrice=10.0,
        productDescription="Sauteed shrimp with cilantro lime rice, avocado, and a spicy chipotle sauce, wrapped in a flour tortilla",
        availableAddOns=[
            Ingredient(name="Roasted Corn Salsa",extra_cost=1.0),
            Ingredient(name="Coleslaw",extra_cost=1.0),
            Ingredient(name="Extra Shrimp",extra_cost=1.0),
            Ingredient(name="Guacamole",extra_cost=1.0)
        ]
    ),
     Product(
        productName="Veggie Burrito",
        productCategory="Burritos",
        productPrice=10.0,
        isVegetarian=True,
        productDescription="Roasted bell peppers, onions, zucchini, and black beans, with rice and Monterey Jack cheese in a whole wheat tortilla",
        availableAddOns=[
            Ingredient(name="Tofu",extra_cost=1.0),
            Ingredient(name="Spinach",extra_cost=1.0),
            Ingredient(name="Pico de Gallo",extra_cost=1.0),
            Ingredient(name="Guacamole",extra_cost=1.0)
        ]
    )
   
]

# Drinks
drink_products = [
    Product(
        productName="Horchata",
        productCategory="Drinks",
        productPrice=3.5,
        productDescription="Traditional Mexican drink made from rice, milk, vanilla, and cinnamon, served chilled",
        isVegetarian=True
    ),
    Product(
        productName="Agua de Jamaica",
        productCategory="Drinks",
        productPrice=3.5,
        productDescription=" Hibiscus iced tea, a sweet and tart drink made from dried hibiscus flowers, garnished with a slice of lime",
        isVegetarian=True
    )
    
]

sweet_products = [
    Product(
        productName="Flan",
        productCategory="Sweets",
        productPrice=4.5,
        productDescription="A creamy caramel custard made with eggs, milk, and sugar, topped with a rich caramel sauce. This dessert is both smooth and sweet, offering a classic finish to any meal",
        isVegetarian=True
    ),
    Product(
        productName="Tres Leches Cake",
        productCategory="Sweets",
        productPrice=3.5,
        productDescription="A light and fluffy sponge cake soaked in a sweet mixture of three kinds of milk: evaporated milk, condensed milk, and heavy cream. It's topped with a thin layer of whipped cream to balance the richness",
        isVegetarian=True
    )
    
]

# Combine all products
mexican_products.extend(taco_products)
mexican_products.extend(burrito_products)
mexican_products.extend(drink_products)
mexican_products.extend(sweet_products)


products_collection = db['products']
products_collection.insert_many([product.model_dump() for product in mexican_products])  


print("Detailed Mexican products inserted successfully!")
