class ShoppingCart(object):
    def __init__(self, product_database):
        # Constructor for the ShoppingCart class.
        # Initializes an empty cart and stores a reference to the product database.
        self.cart = {}  # Dictionary to store products added to the cart
        self.product_database = product_database

    def add_product(self, product_id):
        # Adds a product to the cart by its ID.
        if product_id not in self.cart:
            # If the product is not already in the cart
            product = self.product_database.get_product(product_id)
            if product:
                # If the product is found in the database, add it to the cart
                self.cart[product_id] = product
            else:
                # If the product ID is not found in the database
                print(f"Product with ID {product_id} not found.")
        else:
            # If the product is already in the cart
            print(f"Product with ID {product_id} is already in the cart.")

    def remove_product(self, product_id):
        # Removes a product from the cart by its ID.
        if product_id in self.cart:
            # If the product is in the cart, remove it
            del self.cart[product_id]
        else:
            # If the product ID is not found in the cart
            print(f"Product with ID {product_id} not found in the cart.")

    def clear_cart(self):
        # Clears all products from the cart.
        self.cart.clear()

    def list_products(self):
        # Returns a list of products currently in the cart.
        return self.cart.values()

    def total_price(self):
        # Calculates and returns the total price of all products in the cart.
        return sum(
            product.price if product.price is not None else 0
            for product in self.cart.values()
        )
