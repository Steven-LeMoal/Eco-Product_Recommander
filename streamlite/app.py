import streamlit as st
import json

try:
    from Products_database import ProductDatabase
    from shopping import ShoppingCart
except Exception:
    from streamlite.Products_database import ProductDatabase
    from streamlite.shopping import ShoppingCart

import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from gensim.models import Word2Vec
import random
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


json_file = "data/final_extracted_products.json"
unknown = ["unknown", "not-applicable"]

f = open(json_file)
json_data = json.load(f)

product_database = ProductDatabase(json_data)

# Load Word2Vec model
model = Word2Vec.load("model/ingredients2vec.model")

# Grades scores of the eco-score and nutri-score
grade_scores = {
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "e": 5,
    "f": 6,
    "": 7,
    None: 7,
    "unknown": 7,
}

# Static variable for the website personnalized service
if "cart" not in st.session_state:
    st.session_state.cart = ShoppingCart(product_database)
    st.session_state.total_price = 0
    st.session_state.total_items = 0
if "current_page" not in st.session_state:
    st.session_state.current_page = 1
if "items_per_page" not in st.session_state:
    st.session_state.items_per_page = 10


def generate_cart_json(cart):
    cart_data = {
        product_id: {
            "id": product.id,
            "name": product.name,
            "price": product.price,
            "ecoscore": product.eco_grade,
            "nutriscore": product.nutriscore_grade,
        }
        for product_id, product in cart.cart.items()
    }
    return json.dumps(cart_data, indent=4)


def update_cart_info():
    st.session_state.total_price = st.session_state.cart.total_price()
    st.session_state.total_items = len(st.session_state.cart.cart)


def add_custom_css():
    # CSS for the website
    custom_css = """
    <style>
        /* Custom styles for dropdown menus */
        .stSelectbox .css-2b097c-container { border-color: grey; }

        /* Custom styles for table cells */
        .stMarkdown > table > tbody > tr > td {
            background-color: #f0f0f0;  /* Light grey background */
            color: black;  /* Black text color */
        }

        .stMarkdown > table > thead > tr > th {
            background-color: #f0f0f0;  /* Light grey background for headers */
            color: black;  /* Black text color for headers */
        }

    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # .stApp {
    #     background-color: #f0f2f6;
    # }


def product_embedding(ingredients):
    # Compute the products embeddings given the ingredients embeddings
    valid_ingredients = [
        ingredient for ingredient in ingredients if ingredient in model.wv
    ]
    if not valid_ingredients:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[ingredient] for ingredient in valid_ingredients], axis=0)


def compute_embeddings_and_similarity():
    # Compute embeddings for all products
    embeddings = {
        product_id: product_embedding(product.ingredients)
        for product_id, product in product_database.products.items()
    }

    # Compute similarity matrix (example using cosine similarity)
    embeddings_matrix = np.array(list(embeddings.values()))
    similarity_matrix = cosine_similarity(embeddings_matrix)

    # Store embeddings and similarity matrix in session state
    st.session_state.embeddings_keys = list(embeddings.keys())
    st.session_state.similarity_matrix = similarity_matrix


embeddings_computed = False
if not embeddings_computed and "similarity_matrix" not in st.session_state:
    compute_embeddings_and_similarity()
    embeddings_computed = True


def ngram_similarity(name1, name2):
    """Calculate n-gram similarity between two strings."""
    vectorizer = TfidfVectorizer(min_df=1, analyzer="char", ngram_range=(2, 3))
    tfidf = vectorizer.fit_transform([name1, name2])
    return cos_sim(tfidf)[0, 1]


# -------------------------------- WEBSITE PAGE ------------------------------

# Code for the page : Search product


def page_search_products():
    add_custom_css()
    update_cart_info()

    st.header("Search for Products")
    # Display cart information
    cart_info = f"Total Items in Cart: {st.session_state.total_items}/10 - Total Price: ${st.session_state.total_price}"
    st.markdown(cart_info)
    query = st.text_input("\n\nEnter a product name or property to search:")
    search_results = product_database.search_products_ngram(query)

    if search_results:
        # Add a checkbox to filter for priced products
        show_priced_only = st.checkbox("Show only products with prices")

        if show_priced_only:
            search_results = [
                product
                for product in search_results
                if product.price is not None and product.price > 0.0
            ]

    if search_results:
        # Extracting unique values for each property for filtering
        unique_eco_scores = list(
            set(
                [
                    product.eco_grade if product.eco_grade is not None else "Missing"
                    for product in search_results
                ]
            )
        )

        unique_nutriscores = list(
            set(
                [
                    product.nutriscore_grade
                    if product.nutriscore_grade is not None
                    else "Missing"
                    for product in search_results
                ]
            )
        )

        # Price Range Slider
        min_price = min(
            [
                product.price if product.price is not None else 0
                for product in search_results
            ]
        )
        max_price = max(
            [
                product.price if product.price is not None else 0
                for product in search_results
            ]
        )

        col2, col3, col4 = st.columns([1, 1, 2])

        with col2:
            # Assuming you have a filter for Eco Score
            selected_eco_score = st.selectbox("Eco Score", ["Any"] + unique_eco_scores)
        with col3:
            # Assuming you have a filter for Nutriscore
            selected_nutriscore = st.selectbox(
                "Nutriscore", ["Any"] + unique_nutriscores
            )
        with col4:
            if max_price > min_price:
                price_range = st.slider(
                    "Price Range", min_price, max_price, (min_price, max_price)
                )
            else:
                price_range = None

        if price_range:
            filtered_results = [
                product
                for product in search_results
                if (
                    selected_eco_score
                    in [
                        "Any",
                        product.eco_grade
                        if product.eco_grade is not None
                        else "Missing",
                    ]
                )
                and (
                    selected_nutriscore
                    in [
                        "Any",
                        product.nutriscore_grade
                        if product.nutriscore_grade is not None
                        else "Missing",
                    ]
                )
                and (product.price if product.price is not None else 0)
                >= price_range[0]
                and (product.price if product.price is not None else 0)
                <= price_range[1]
            ]
        else:
            filtered_results = [
                product
                for product in search_results
                if (
                    selected_eco_score
                    in [
                        "Any",
                        product.eco_grade
                        if product.eco_grade is not None
                        else "Missing",
                    ]
                )
                and (
                    selected_nutriscore
                    in [
                        "Any",
                        product.nutriscore_grade
                        if product.nutriscore_grade is not None
                        else "Missing",
                    ]
                )
            ]

        # Pagination
        total_items = len(filtered_results)
        total_pages = total_items // st.session_state.items_per_page + (
            total_items % st.session_state.items_per_page > 0
        )
        st.session_state.current_page = min(
            st.session_state.current_page, total_pages
        )  # Adjust current page if out of range

        start_idx = (
            st.session_state.current_page - 1
        ) * st.session_state.items_per_page
        end_idx = start_idx + st.session_state.items_per_page
        displayed_products = filtered_results[start_idx:end_idx]

        header_cols = st.columns([9, 2, 2, 2, 1])
        headers = ["Name", "Price", "Eco Score", "Nutriscore", "Add"]
        for col, header in zip(header_cols, headers):
            col.write(header)

        for product in displayed_products:
            # Create a row for each product
            col_name, col_price, col_eco, col_nutri, col_add = st.columns(
                [9, 2, 2, 2, 1]
            )

            with col_name:
                expander = st.expander(f"{product.name}")
            with col_price:
                st.text(f"${product.price}" if product.price else "N/A")
            with col_eco:
                st.text(
                    product.eco_grade
                    if product.eco_grade and product.eco_grade not in unknown
                    else "N/A"
                )
            with col_nutri:
                st.text(
                    product.nutriscore_grade
                    if product.nutriscore_grade
                    and product.nutriscore_grade not in unknown
                    else "N/A"
                )
            with col_add:
                if st.button(" + ", key=f"add_{product.id}"):
                    st.session_state.cart.add_product(product.id)
                    update_cart_info()
                    st.experimental_rerun()
                elif len(st.session_state.cart.cart) >= 10:
                    st.write("Maximun of 10 products in cart reached.")

            with expander:
                st.markdown(product.format_product_details())

        # Pagination controls
        col_prev, col_page_info, col_next = st.columns([4.5, 4.5, 1])
        with col_prev:
            if st.button("Previous") and st.session_state.current_page > 1:
                st.session_state.current_page -= 1
                st.experimental_rerun()
        with col_page_info:
            st.write(f"Page {st.session_state.current_page} of {total_pages}")
        with col_next:
            if st.button("Next") and st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
                st.experimental_rerun()
    else:
        st.write("No products found.")


# Code for the page : Revised cart (in the page Cart)


def display_revised_cart():
    st.header("Revised Cart")
    # Display products in the revised cart

    header_cols = st.columns([9, 2, 2, 2])
    headers = ["Name", "Price", "Eco Score", "Nutriscore"]
    for col, header in zip(header_cols, headers):
        col.write(header)

    # Rows for each product
    for product_id, product in st.session_state.revised_cart.cart.items():
        col_name, col_price, col_eco, col_nutri = st.columns([9, 2, 2, 2])

        with col_name:
            expander = st.expander(f"{product.name}")
        with col_price:
            st.text(f"${product.price}" if product.price else "N/A")
        with col_eco:
            st.text(
                product.eco_grade
                if product.eco_grade and product.eco_grade not in unknown
                else "N/A"
            )
        with col_nutri:
            st.text(
                product.nutriscore_grade
                if product.nutriscore_grade and product.nutriscore_grade not in unknown
                else "N/A"
            )

        with expander:
            st.markdown(product.format_product_details())

    # st.write(
    #     f"Total Items in Cart : {st.session_state.total_items}/10 - Total Price: ${st.session_state.total_price}"
    # )

    if st.button("Use Revised Cart"):
        st.session_state.cart.cart = st.session_state.revised_cart

        update_cart_info()
        st.experimental_rerun()


def meets_criteria(product, option, current_grade):
    # Define a scoring system for the grades

    if option == "Make Vegan":
        return "vegan" in product.vegetarian
    elif option == "Make Vegetarian":
        return "vegetarian" in product.vegetarian
    elif option == "Best Eco Grade":
        product_grade = product.eco_grade if product.eco_grade in grade_scores else ""
        return grade_scores[product_grade] < grade_scores[current_grade.eco_grade]
    elif option == "Best Nutri Score":
        product_grade = (
            product.nutriscore_grade if product.nutriscore_grade in grade_scores else ""
        )
        return (
            grade_scores[product_grade] < grade_scores[current_grade.nutriscore_grade]
        )


def find_similar_product(current_product_id, option, alpha, similarity_threshold):
    # Retrieve the index of the current product in the similarity matrix
    current_index = st.session_state.embeddings_keys.index(current_product_id)

    highest_similarity = 0
    most_similar_product_id = current_product_id  # Default to current product
    curr_product = product_database.get_product(current_product_id)
    current_name = curr_product.name

    grade = "Best" in option

    for product_id in st.session_state.embeddings_keys:
        if product_id == current_product_id:
            continue  # Skip the same product

        # Retrieve the index of the other product
        other_index = st.session_state.embeddings_keys.index(product_id)

        # Get the similarity score from the precomputed matrix
        similarity = st.session_state.similarity_matrix[current_index][other_index]

        other = product_database.get_product(product_id)

        name_similarity = 0
        if current_name and other.name:
            name_similarity = ngram_similarity(current_name, other.name)

        # Combined score: alpha * name_similarity + (1 - alpha) * embedding_similarity
        combined_score = alpha * name_similarity + (1 - alpha) * similarity

        if (
            combined_score > highest_similarity
            and combined_score >= similarity_threshold
        ):
            # Check if the product meets the selected criteria
            if meets_criteria(
                other,
                option,
                None if not grade else curr_product,
            ):
                highest_similarity = combined_score
                most_similar_product_id = product_id

    return most_similar_product_id


def revise_cart(option, alpha, similarity_threshold):
    revised_cart = ShoppingCart(product_database)

    for product_id in st.session_state.cart.cart.keys():
        # Find the most similar product based on embeddings and criteria
        similar_product_id = find_similar_product(
            product_id, option, alpha, similarity_threshold
        )
        revised_cart.add_product(similar_product_id)

    # Store the revised cart in the session state
    st.session_state.revised_cart = revised_cart
    display_revised_cart()


# Code for the page : View cart


def page_view_cart():
    add_custom_css()
    st.header("Shopping Cart")
    if st.session_state.cart.cart:
        header_cols = st.columns([9, 2, 2, 2, 1])
        headers = ["Name", "Price", "Eco Score", "Nutriscore", ""]
        for col, header in zip(header_cols, headers):
            col.write(header)

        # Rows for each product
        for product_id, product in st.session_state.cart.cart.items():
            col_name, col_price, col_eco, col_nutri, col_remove = st.columns(
                [9, 2, 2, 2, 1]
            )

            with col_name:
                expander = st.expander(f"{product.name}")
            with col_price:
                st.text(f"${product.price}" if product.price else "N/A")
            with col_eco:
                st.text(
                    product.eco_grade
                    if product.eco_grade and product.eco_grade not in unknown
                    else "N/A"
                )
            with col_nutri:
                st.text(
                    product.nutriscore_grade
                    if product.nutriscore_grade
                    and product.nutriscore_grade not in unknown
                    else "N/A"
                )
            with col_remove:
                if st.button(" - ", key=f"remove_{product.id}"):
                    st.session_state.cart.remove_product(str(product_id))
                    update_cart_info()
                    st.experimental_rerun()

            with expander:
                st.markdown(product.format_product_details())

        col1_btn, col2_btn = st.columns([9.2, 1.5])

        with col1_btn:
            if st.button("Clear Cart"):
                st.session_state.cart.clear_cart()
                update_cart_info()
                st.experimental_rerun()
        with col2_btn:
            if st.session_state.cart.cart:
                json_cart_data = generate_cart_json(st.session_state.cart)
                st.download_button(
                    label="Download Cart",
                    data=json_cart_data,
                    file_name="cart.json",
                    mime="application/json",
                )
        st.write(
            f"Total Items in Cart : {st.session_state.total_items}/10 - Total Price: ${st.session_state.total_price}"
        )

        st.header("Revise Your Cart")
        revise_option = st.selectbox(
            "Choose your revision criteria:",
            ("Make Vegan", "Make Vegetarian", "Best Eco Grade", "Best Nutri Score"),
        )
        alpha = st.slider("Weight for Name Similarity (alpha)", 0.0, 1.0, 0.5)
        similarity_threshold = st.slider("Minimum similarity threshold", 0.0, 1.0, 0.5)
        if st.button("Revise Cart"):
            revise_cart(revise_option, alpha, similarity_threshold)

    else:
        st.write("Your cart is empty.")


# Code for the page : Welcome page


def page_welcome():
    st.title("Welcome to Eco-Product Recommender!")
    # st.image("path_to_your_welcome_image.jpg", use_column_width=True)
    st.markdown(
        """
    ### Making Healthier and Sustainable Food Choices Simplified

    Our tool helps you compare food products based on nutritional and environmental criteria,
    assisting you in making informed decisions for a healthier lifestyle and a better planet.

    ---

    **Get started by navigating to the _Search_ page to find products or go to your _Cart_ to view selected items.**
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)  # Add some space

    # Optional: Add an expander for more detailed introduction or guide
    with st.expander("Learn more about how to use this tool"):
        st.write(
            """
            - First go to the products search page
            - Select the products you want, by adding them to the cart
            - Go to the cart page
            - Refine your product selection
            ...
            """
        )
    st.markdown(
        "This project is only for educationnal purposis and the data used (price) will not be made public"
    )

    # image = Image.open("open_food_fact.png")


# Code for the page : Embeddings visualisation


def product_embedding_3d_page():
    st.title("Product Embedding Visualization")

    # Slider for selecting the percentage of products
    percentage = st.slider(
        "Select the percentage of products for visualization", 0, 100, 10
    )

    # User selection for coloring points

    color_by = st.selectbox(
        "Color points by",
        ["Eco Grade", "Nutriscore Grade", "Vegetarian Status", "Vegan Status"],
    )

    # Function to determine color based on the selected attribute
    def get_color(product):
        if color_by == "Eco Grade":
            return eco_grade_color_mapping.get(product.eco_grade, default_color)
        elif color_by == "Nutriscore Grade":
            return nutriscore_color_mapping.get(product.nutriscore_grade, default_color)
        elif color_by == "Vegetarian Status":
            if "vegetarian" in product.vegetarian:
                return vegetarian_color["vegetarian"]
            elif "non vegetarian" in product.vegetarian:
                return vegetarian_color["non vegetarian"]
            else:
                return default_color
        elif color_by == "Vegan Status":
            if "vegan" in product.vegetarian:
                return vegan_color["vegan"]
            elif "non vegan" in product.vegetarian:
                return vegan_color["non vegan"]
            else:
                return default_color

    # Color mappings (customize as needed)
    eco_grade_color_mapping = {
        "a": "green",
        "b": "blue",
        "c": "yellow",
        "d": "orange",
        "e": "red",
        "f": "purple",
    }
    nutriscore_color_mapping = {
        "a": "green",
        "b": "blue",
        "c": "yellow",
        "d": "orange",
        "e": "red",
        "f": "purple",
    }

    default_color = "gray"

    vegan_color = {
        "vegan": "lightblue",
        "non vegan": "red",
    }

    vegetarian_color = {
        "vegetarian": "lightblue",
        "non vegetarian": "blue",
    }

    color_mappings = (
        eco_grade_color_mapping
        if color_by == "Eco Grade"
        else nutriscore_color_mapping
        if color_by == "Nutriscore Grade"
        else vegan_color
        if color_by == "Vegan Status"
        else vegetarian_color
    )

    if (
        "percentage" not in st.session_state
        or st.session_state.percentage != percentage
    ):
        st.session_state.percentage = percentage

        # Randomly select a percentage of products
        num_products = len(json_data)
        num_selected = int(num_products * (percentage / 100))
        selected_products = random.sample(list(json_data.values()), num_selected)

        # Compute embeddings for the selected products
        selected_embeddings = [
            product_embedding(product["ingredients_tags"])
            for product in selected_products
        ]

        # Convert embeddings to a format suitable for visualization
        embedding_matrix = np.array(selected_embeddings)

        # Use t-SNE for dimensionality reduction to 3 components
        tsne = TSNE(n_components=3, random_state=1236)
        reduced_embeddings = tsne.fit_transform(embedding_matrix)

        # ... [Code to compute embeddings based on the selected percentage] ...
        # Update session state
        st.session_state.embeddings = reduced_embeddings
        st.session_state.selected_products = selected_products

        # Assign colors to each product based on the selected attribute
    colors = [
        get_color(product_database.get_product(product["_id"]))
        for product in st.session_state.selected_products
    ]

    # Create a list to hold traces
    data = []

    scatter_trace = go.Scatter3d(
        x=st.session_state.embeddings[:, 0],
        y=st.session_state.embeddings[:, 1],
        z=st.session_state.embeddings[:, 2],
        mode="markers",
        marker=dict(size=5, opacity=0.8, color=colors),  # Use the computed colors
        text=[
            (f"{product_database.get_product(product['_id'])}").replace("\n", "<br>")
            for product in st.session_state.selected_products
        ],
        hoverinfo="text",
        name="Products",
    )

    data.append(scatter_trace)

    # Add invisible traces for the legend
    for label, color in color_mappings.items():
        legend_trace = go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],  # No actual data
            mode="markers",
            marker=dict(size=10, color=color),
            name=label,  # Label for the legend
        )
        data.append(legend_trace)

    # Define the layout and figure
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0), title="3D Product Embeddings", showlegend=True
    )
    fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig, use_container_width=True)


st.set_page_config(
    page_title="Eco-Product Recommender",
    page_icon="🌿",  # You can use an emoji or an image as an icon
    layout="wide",  # Use the wide layout mode for more space
    initial_sidebar_state="expanded",  # Keep the sidebar expanded
)

# Add the new page to the navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ("Welcome", "Search Products", "View Cart", "Product Embedding Visualization"),
)

if page == "Welcome":
    page_welcome()
elif page == "Search Products":
    page_search_products()
elif page == "View Cart":
    page_view_cart()
elif page == "Product Embedding Visualization":
    product_embedding_3d_page()
# Run this script with `streamlit run app.py`
