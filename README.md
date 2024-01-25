# Eco-Recommend Product Recommendation App

## Overview
Eco-Recommend is a Streamlit-based web application designed to assist users in making environmentally responsible and health-conscious purchasing decisions. It leverages product data to recommend items based on ecological and nutritional scores, and offers features like product search, cart management, and eco-friendly product suggestions.
This project is only for educationnal purposis only and the data used (price) will/should not be made public or be used in any way others than for this demonstration project.

**Group (srapping & website):** Natacha Batmini Batmanabane; Steven Le Moal

**New Project:**
- **Question:** What are the best food products according to my nutritional and environmental criteria? How to compare prices and nutritional quality of food products?
- **Objective:** To provide a personalized selection of food products, based on specific criteria such as nutritional value, cost, and environmental impact, by analyzing and comparing information available on targeted websites.

**Sites to Scrape:**
- `https://www.carrefour.fr/s?q={nom_nourriture}` (data in the GitHub)
- `https://fr.openfoodfacts.org/cgi/search.pl?search_terms={nom_nourriture}&search_simple=1&action=process` (download the JSON file)
- JSON generated using the API/request of OpenFoodFact: [Google Drive Link](https://drive.google.com/file/d/1-iRCuS0x9eZOWKCoifC2TDhl44qyP3EK/view?usp=sharing)

## Features
- **Product Search**: Search for products based on names or attributes. Supports filtering by eco-scores, nutriscore-scores, and price ranges.
- **Shopping Cart**: Add and remove products from a shopping cart. View total price and item count.
- **Eco-Friendly Alternatives**: Suggests alternative products with better eco-scores or healthier options.
- **3D Product Embedding Visualization (Ingredients2vec)**: Explore products in a 3D space based on their ingredient embeddings. Interact with the plot to learn more about each product.
- **Cart Revision Options**: Revise the cart to make it vegan, vegetarian, or to optimize for the best eco or nutri-scores.

## NLP
Please refer to the notebook (nlp) for more details or to this presentation : https://docs.google.com/presentation/d/1OG3j9HyNXKs2Fm7S1vtEgHyg1kjnJLevY0DeOD8iKH8/edit?usp=sharing

For better visualisation, please check the nlp notebook :
You can launch tensorboard on logs or logs_tfidf then go to projector (then use TSNE and put supervision > 0)
`tensorboard --logdir=logs_tfidf`

## Installation and Setup
1. **Clone the Repository**:  
   `git clone https://github.com/Steven-LeMoal/Eco-Product_Recommander.git`
2. **Load Data**:
   Please download the data and put the files in the data folder : https://drive.google.com/drive/folders/1dqSfEpSFdDmLPGXw4rHxPSA2ODhidTrl?usp=sharing
   (only final_extracted_products.json is necessary to run the streamlit app)
   Place your product JSON data in the designated data directory.
3. **Run the Application**:  
   Run the Streamlit app, you can directly use the notebook (dbs_and_streamlite_use):  
   `streamlit run app.py`

## Folder Structure
```
Project
│   app.py
│   final_extracted_products.json
│
├───data
│       Prices.csv
│       PRICE_subset.csv
│       data.csv
│       final_extracted_products.json (only necessary for streamlite app)
│       PRICE_subset.json
│       all_veg_data.csv
│
├───logs
│       embedding_tfidf.ckpt-1.data-00000-of-00001
│       embedding.ckpt-1.index
│       ...
│
├───model
│       ingredients2vec.model
│
├───notebook
│       full_scrapping.ipynb
│       scrapping_food_fact.ipynb
│       nlp.ipynb
│       dbs_and_streamlite_use.ipynb
│
└───streamlite
    │   Products_database.py
    │   shopping.py
    │   Products.py
    │   open_food_fact.png
    │   ...
```

## Contributing
Contributions to Eco-Recommend are welcome! Please read our contributing guidelines to get started.

## License
[MIT License](LICENSE)

## Contact
For any queries or suggestions, please reach out to us at [stevenlemoal-@outlook.fr].
