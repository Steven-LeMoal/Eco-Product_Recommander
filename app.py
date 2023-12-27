import streamlit as st
import json
from Products_database import ProductDatabase
from shopping import ShoppingCart

json_file = "final_extracted_products.json"
unknown = ["unknown", "not-applicable"]

f = open(json_file)
json_data = json.load(f)

product_database = ProductDatabase(json_data)

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

    else:
        st.write("Your cart is empty.")


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


st.set_page_config(
    page_title="Eco-Product Recommender",
    page_icon="🌿",  # You can use an emoji or an image as an icon
    layout="wide",  # Use the wide layout mode for more space
    initial_sidebar_state="expanded",  # Keep the sidebar expanded
)

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Welcome", "Search Products", "View Cart"))

if page == "Welcome":
    page_welcome()
if page == "Search Products":
    page_search_products()
elif page == "View Cart":
    page_view_cart()

# Run this script with `streamlit run app.py`
