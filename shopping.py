class ShoppingCart(object):
    def __init__(self, product_database):
        self.cart = {}
        self.product_database = product_database

    def add_product(self, product_id):
        if product_id not in self.cart:
            product = self.product_database.get_product(product_id)
            if product:
                self.cart[product_id] = product
            else:
                print(f"Product with ID {product_id} not found.")
        else:
            print(f"Product with ID {product_id} is already in the cart.")

    def remove_product(self, product_id):
        if product_id in self.cart:
            del self.cart[product_id]
        else:
            print(f"Product with ID {product_id} not found in the cart.")

    def clear_cart(self):
        self.cart.clear()

    def list_products(self):
        return self.cart.values()

    def total_price(self):
        return sum(product.price if product.price is not None else 0 for product in self.cart.values())
