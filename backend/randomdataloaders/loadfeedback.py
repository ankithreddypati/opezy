import os
import pymongo
import certifi
import random
from datetime import datetime, timedelta
from faker import Faker
from dotenv import load_dotenv
from pymongo import UpdateOne
from pymongo.errors import ServerSelectionTimeoutError

from models import Feedback

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

#Random feedback using AI to tell a story from feedback
feedback_entries = [
    "The Taco al Pastor was a delightful surprise! Juicy pork with a perfect smoky flavor. Definitely coming back for more.",
    "Loved the cozy vibe at this place. It’s not just the food, but the warm lighting and Mexican décor that make it special.",
    "Barbacoa Tacos were incredibly tender and flavorful. The salsa on top brought a nice heat.",
    "Service was top-notch. Our waiter was attentive and gave great recommendations, particularly the Pollo Asado Taco.",
    "Fish Taco lacked a bit in flavor. Could use a more vibrant sauce to kick it up a notch.",
    "Chicken Burrito was huge and stuffed with goodness! So satisfying and worth every penny.",
    "Had the Breakfast Burrito this morning. It was the perfect start to my day with just the right amount of spice.",
    "Shrimp Burrito was a letdown. It felt a bit too greasy and the shrimp werent as fresh as I’d hoped.",
    "The Veggie Burrito was a hit among my vegetarian friends. Fresh ingredients and lots of flavors.",
    "Horchata was the perfect sweet treat to end our meal. Creamy and not too sweet, just how I like it.",
    "Flan was absolutely divine! Silky smooth with a caramel sauce that was just heavenly.",
    "Tres Leches Cake was a bit too soggy for my taste. Might skip dessert next time.",
    "The ambiance transports you straight to Mexico. Colorful, lively, and absolutely inviting.",
    "Music was a tad too loud tonight. Made it hard to have a conversation at our table.",
    "I appreciate the cleanliness of this place. It always feels fresh and well-maintained.",
    "Wait times can be long during peak hours, but the food is definitely worth the wait.",
    "Pollo Asado Taco had a wonderful charred flavor. The marinade was spot-on.",
    "Servers were friendly but seemed overwhelmed. It took a while for our food to arrive.",
    "Outdoor seating area is lovely. Great for enjoying a sunny day and some delicious tacos.",
    "The Margaritas here are the best I've had! Perfectly balanced and not too sweet.",
    "Servers did a fantastic job explaining the different types of tacos. Great service!",
    "The salsa bar is amazing! Love the variety and the fresh flavors.",
    "Very impressed with the cleanliness of the bathrooms. Shows they care about all aspects of customer service.",
    "The Tortilla Soup had a rich, authentic flavor. Warmed me up on a chilly evening.",
    "Fish Tacos were a bit underwhelming this time—needed a bit more seasoning.",
    "Absolutely loved the Ceviche! Fresh, zesty, and full of flavor.",
    "Parking was a breeze. Plenty of space and well-lit at night.",
    "The Carnitas are a must-try. So tender and juicy!",
    "Please hire more waiters to run the business",
    "Enjoyed the live band they had last night. Added a fun touch to our dining experience.",
    "Had to wait a bit too long for our check when we were ready to leave.",
    "The enchiladas are just like my abuela used to make. So authentic and tasty.",
    "Outdoor patio is wonderful for dining al fresco. Really enjoyed the atmosphere.",
    "The Guacamole was freshly made at our table. Delicious and entertaining to watch.",
    "Their spicy Margarita has a real kick to it. Not for the faint-hearted!",
    "The staff was very accommodating of our large group. Made our celebration special.",
    "Loved the variety in the taco platter. Great for sharing and trying different flavors.",
    "Could use more vegan options. The choices were somewhat limited.",
    "The Nachos were a bit too soggy for my liking. Needed a crispier touch.",
    "The restaurant was a bit too cold. Needed to keep my jacket on while eating.",
    "Huevos Rancheros for breakfast was a delightful experience. Highly recommend.",
    "The Queso Fundido is a dream for cheese lovers. So gooly and rich.",
    "Their house-made hot sauces are incredible. Bought a couple of bottles to take home!",
    "The decor is charming, filled with Mexican crafts and vibrant colors.",
    "Waiter was a bit pushy with upselling more expensive items.",
    "Celebrated our anniversary here. The staff made it very special with complimentary dessert.",
    "The Mole sauce on the chicken was a bit too thick. Flavor was good, though.",
    "Loved the vegetarian chili. It was hearty and packed with flavor.",
    "The restaurant has a nice family-friendly vibe. Feels welcoming to all ages.",
    "No alcohol please, No one is liking the vibe here recently",
    "Service was quick and efficient. We were in and out in no time.",
    "The steak in the burrito was perfectly cooked. So tender and flavorful.",
    "Would love to see more seafood options on the menu.",
    "Hire more people to handle the rush hours",
    "The spicy chicken wings were a great start to our meal. Just the right amount of heat.",
    "Felt the music selection didn't quite match the ambiance of the restaurant.",
    "The rice and beans side dish was surprisingly good—flavorful and comforting.",
    "Appreciate that they source their ingredients locally. You can taste the freshness.",
    "The street corn was absolutely delicious, with just the right amount of cheese and spice.",
    "Had a lovely time on the terrace. It’s beautifully lit and quite romantic in the evening.",
    "Your one day promotion of Jalapeño Poppers had a great crispy coating and a fiery kick. Highly recommend to get it back",
    "Portion sizes are generous. You definitely get good value for your money here.",
    "The staff's attire, adorned with traditional Mexican elements, added to the authentic experience.",
    "The flan is a must-try dessert. Unique and incredibly tasty.",
    "The mariachi band on weekends was very good i would love to see more of it ",
    "Can you open little early in the morning",
    "I was really surprised to see alcohol on the menu. I preferred when this place was family-friendly and alcohol-free.",
    "Adding alcohol to the menu has really changed the vibe here. It used to be my go-to spot for quiet evenings, but not anymore.",
    "Found the dining chairs a bit uncomfortable. Could do with some cushions.",
    "The salsa verde had a fantastic depth of flavor. Went well with everything!",
    "Service was a tad slow, but the staff was very polite and apologetic about the wait.",
    "The Fist taco was a bit too salty for my taste. Usually, it’s spot on.",
    "The restaurant does a great Taco Tuesday deal. Great variety and excellent prices.",
    "Wish the menu had more descriptions for those unfamiliar with Mexican cuisine.",
    "Not thrilled about the introduction of alcohol. I chose this place for its sober environment, and it's disappointing to see that change.",
    "The tacos were perfectly crispy and topped with just the right amount of cheese.",
    "I've noticed a shift in the clientele and the atmosphere since you started serving alcohol. It’s not the relaxed, sober space it used to be.",
    "Parking can be challenging during peak hours. Best to come early or use public transport.",
    "Long waiting times you can hire more employees",
    "Please try to open the store early in the morning",
    "The quesadilla was packed with cheese and flavorful mushrooms. Very satisfying.",
    "They could improve the lighting over the tables. It was a bit dim for reading the menu.",
    "The pork carnitas were a bit dry today. Usually, they're much juicier.",
    "Loved the artwork on the walls. Gives a great insight into Mexican culture.",
    "Their spicy chocolate cake is an unexpected delight. Rich with a hint of chili.",
    "I used to bring my kids here all the time because it was alcohol-free. I’m not sure we’ll come as often now that’s changed.",
    "Bringing alcohol was a letdown for me. It’s harder to enjoy my meals with the bar scene picking up.",
    "The kids' play area is a nice touch. Makes it a great spot for families.",
    "The complimentary chips and salsa upon arrival are always a welcome treat.",
    "Recent introduction of alcohol section made the restaurant horrible ",
    "I dont want the alcohol here. It does not suit your style im sorry",
    "The guacmole seems not fresh",
    "I was a regular because this was a safe space away from alcohol. I'm concerned this new direction might change things too much for my liking.",
    "The tacos were authentic and packed with flavor. Perfect for sharing.",
    "The guacamole was not up to par today. It had a mushy texture and a slightly brown color, which wasn't very appetizing.",
    "The flan dessert was soggy. didnot like it much",
    "Unfortunately, the guacamole didn't meet my expectations this time. It seemed a bit brown and wasn't as vibrant in flavor or color as I hoped.",
    "Today's guacamole didn't seem fresh. It had a browner appearance and lacked the creamy, fresh avocado flavor I’ve enjoyed in the past.",
    "we liked the restaurant when there was only non-alcoholic drinks, Bringing the alchol killed the vibe of the restaurant ",
    "Enjoyed the relaxed pace of dining here. It's never rushed, always enjoyable.",
    "The waiter mixed up our order but resolved the issue swiftly and with a smile.",
    "The vegan taco options were surprisingly good. Full of flavor and variety.",
    "Im not liking the introdcution of alcohol recently",
    "I’m not happy with the guacamole I received today. It had a stale taste and the color was dull, not the vibrant green I'm accustomed to here.",
    "Could do with more spice options. I like my food hotter.",
    "The live mariachi band on weekends adds a fantastic, authentic vibe to the place.",
    "Not sure what happened, but the guacamole was a letdown. It was darker than normal and the flavor was not as bright and tangy as before.",
    "Appreciated the attention to detail in the presentation of the plates.",
    "The tortillas are a standout. You can really taste the difference.",
    "The Horchata is a perfect drink here.",
    "The introduction of alcohol seems to contradict what I loved most about this place—its focus on healthy, mindful eating without the bar scene.",
    "The brunch menu is excellent, especially the Mexican-style eggs Benedict.",
    "The waitstaff are always smiling and genuinely seem to enjoy their jobs .",
    "The chicken here is always tender and flavorful, cooked to perfection.",
    "Disappointed to see alcohol here now. It really changes the environment and not for the better. I miss the old, peaceful setting.",
    "This place is busy at time can hire more people ",
    "I’m not comfortable with the introduction of alcohol. It was better when it was simpler and more inclusive for all ages.",
    "I disliked the veggie burrito dish, the bell peppers in it tasted stale and seemed past their prime.",
    "I disliked the veggie burrito dish as the bell peppers in it had a wrinkled skin and lacked the usual crispness.",
    "I disliked  zucchini as it had a mushy texture and tasted like it had been sitting for too long.",
    "I disliked zucchini in the veggie burrito dish as the slices were discolored and had a slightly off-putting smell.",
    "I disliked guacamole the veggie burrito dish as it had turned brown and had a sour taste, indicating it was old.",
    "I disliked  bell peppers in the veggie burrito dish as they were limp and lacked the crunch that fresh ones usually have.",
    "I disliked  bell peppers in the veggie burrito dish as it had started to shrivel, indicating they were not fresh.",
    "I disliked zucchini in the veggie burrito dish as it had a bitter taste, a sign that it was no longer fresh.",
    "I disliked zucchini in the veggie burrito dish as it had soft spots and felt mushy to the touch.",
    "I disliked the guacamole in the veggie burrito dish as it had separated, with a layer of liquid on top, suggesting it was old.",
    "I disliked  bell peppers in the veggie burrito dish as it had a stale aroma and lacked the usual sweetness.",
    "I disliked zucchini in the veggie burrito dish as it had a rubbery texture and tasted stale.",
    "I disliked zucchini in the veggie burrito dish as it had a dull appearance and lacked the usual crispness.",
    "I did not like guacamole in the veggie burrito dish as it had a fermented taste, indicating it had been left out for too long.",
    "I dislike the zucchini in the veggie burrito dish as it had a foul odor and tasted rancid.",
    "I did not like the bell peppers in the veggie burrito dish ,they had a wilted appearance and tasted bitter.",
    "I dislike that bell peppers in the veggie burrito dish were soft to the touch and had a stale flavor."   
]

def insert_feedback_entries(feedback_entries):
    feedback_list = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    faker = Faker()

    try:
        # Generate feedback entries
        for idx, comment in enumerate(feedback_entries):
            feedback_entry = Feedback(
                feedbackId=str(idx + 1),  
                feedbackComment=comment,
                feedbackDate=faker.date_time_between(start_date=start_date, end_date=end_date)
            )
            feedback_list.append(feedback_entry.model_dump(by_alias=True)) 

        db.feedbacks.bulk_write([
            UpdateOne({"feedbackId": entry["feedbackId"]}, {"$set": entry}, upsert=True)
            for entry in feedback_list
        ])
        
        print(f"Inserted {len(feedback_list)} feedback entries successfully.")
        
    except Exception as e:
        print(f"Error inserting feedback entries: {e}")

insert_feedback_entries(feedback_entries)


print("Feedback data generated and inserted successfully!")