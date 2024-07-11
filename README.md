# Opezy: AI Co-pilot for Small Businesses

## Introduction

**Opezy** is an assistant designed to help small business owners make data-driven decisions and manage various business operations efficiently. This project demonstrates the implementation of Opezy for a Mexican restaurant, showcasing its capabilities in sales analysis, inventory management, customer feedback analysis, marketing strategy generation, and expense monitoring.

<iframe width="560" height="315" src="https://www.youtube.com/embed/C4sH4Qgx83o" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


## Features

1. **Sales Analysis:**
   - Analyze sales trends.
   - Identify best-selling and underperforming products.

2. **Inventory Management:**
   - Monitor inventory freshness using GPT-4 vision.
   - Make informed restocking decisions.

3. **Customer Feedback Analysis:**
   - Collect and analyze customer feedback.
   - Improve products and services based on insights.

4. **Marketing Strategy Generation:**
   - Generate marketing strategies.
   - Create promotional content for social media.

5. **Expense Monitoring:**
   - Detect anomalies in expenses.
   - Prevent fraud and manage costs effectively.

6. **Automated Communications:**
   - Automate sending marketing emails.
   - Manage customer communications seamlessly.

## Technologies Used

- **Azure OpenAI:** For intelligent data analysis and natural language processing.
- **Azure Cosmos DB for MongoDB:** For storing sales, inventory, customer feedback, expense data, and vector embeddings.
- **LangChain:** For developing conversational agents and integrating various tools.
- **React:** For building the user interface.

## Project Structure

- **Backend:** Handles data processing, analysis, and communication with Azure services.
- **Frontend:** Provides a user interface for interacting with Opezy, including a help section and onboarding process.

## Databases Used


To generate data 
 ```sh
cd randomdataloaders
run these to generate random data
 ```

1. **Sales Database:** Tracks all sales transactions, allowing detailed sales analysis.
2. **Inventory Database:** Manages inventory levels and monitors the freshness of products.
3. **Feedback Database:** Collects and analyzes customer feedback to improve products and services.
4. **Expenses Database:** Logs all business expenses and helps in detecting anomalies and managing costs.
5. **Product Database:** Stores product information and ingredients.
6. **Customer Database:** Stores information about customers and their emails.

## Setup and Installation

### Prerequisites

- Node.js
- MongoDB
- Azure Subscription
- Azure Cosmos DB for MongoDB
- Azure OpenAI

### Backend Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/opezy.git
   cd opezy/backend
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Azure and MongoDB credentials check .envsample:
   ```sh
   DB_CONNECTION_STRING=<your_mongo_db_connection_string>
   AOAI_ENDPOINT=<your_azure_openai_endpoint>
   AOAI_KEY=<your_azure_openai_key>
   ```

4. Run the backend server:
   ```sh
   uvicorn --host "0.0.0.0" --port 8000 app:app --reload
   ```

### Frontend Setup
https://github.com/ankithreddypati/opezy-frontend

cd opezy-frontend 

npm start


##Deployement 

cd deploy

Open the `azuredeploy.parameters.json` file, then edit the `mongoDbPassword` to a password you wish to use for the MongoDB Admin User:


When the Azure Bicep template is deployed, this parameters file will be used to configure the Mongo DB Password and other parameters when provisioning the Azure resources.

## Login to Azure

Open a terminal window and log in to Azure using the following command:

```Powershell
Connect-AzAccount
```

### Set the desired subscription (Optional)

If you have more than one subscription associated with your account, set the desired subscription using the following command:

```Powershell
Set-AzContext -SubscriptionId <subscription-id>
```

## Create resource group

```Powershell
New-AzResourceGroup -Name mongo-devguide-rg -Location 'eastus'
```

## Deploy using bicep template

Deploy the solution resources using the following command (this will take a few minutes to run):

```Powershell
New-AzResourceGroupDeployment -ResourceGroupName mongo-devguide-rg -TemplateFile .\azuredeploy.bicep -TemplateParameterFile .\azuredeploy.parameters.json -c
```


## Usage

- **Help Section:** Provides an overview of Opezy's features and how to use them.
- **Onboarding Process:** Allows users to integrate their data and start using Opezy.

### Example Interactions

1. **Analyze Sales Trends:**
   - Ask Opezy: "Which products' sales are declining from January 2024 to March 2024?"
   
2. **Monitor Inventory Freshness:**
   - Ask Opezy: "Can you check how fresh the bell peppers are from the inventory?"

3. **Generate Marketing Strategy:**
   - Ask Opezy: "Give me a marketing strategy to sell more Veggie Burritos."

4. **Detect Expense Anomalies:**
   - Ask Opezy: "Can you detect any anomalies in the expenses from December 2023 to May 2024?"

## Contribution

We welcome contributions to improve Opezy. To contribute, follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.




