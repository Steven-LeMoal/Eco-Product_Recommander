class Product(object):
    """
    A class to represent a product.
    """

    def __init__(self, product_data):
        """
        Initialize the Product instance with product data from the JSON file.
        """
        self.id = product_data.get("_id")

        self.name = product_data.get("product_name")
        self.quantity = product_data.get("quantity")

        # self.url = product_data.get("ecoscore_extended_data", {"Nothing": None}).get(
        #     "url"
        # )

        self.price = None

        self.agribalyse = product_data.get("agribalyse")
        self.eco_score = product_data.get("ecoscore_score")
        self.eco_grade = product_data.get("ecoscore_grade")

        self.nutriscore = product_data.get("nutriscore_scor_data", {})
        self.nutriscore_grade = product_data.get("nutriscore_grade")
        self.ingredients = product_data.get("ingredients_tags", [" ", " "])

    def __str__(self):
        """
        String representation of the product.
        """
        return f"Product(ID: {self.id}, Name: {self.name}, Price: {self.price}, Eco Score: {self.eco_grade}, Nutriscore Score: {self.nutriscore_grade})"

    def format_product_details(self):
        details = ""
        details += f"**ID:** {self.id}\n\n"
        details += f"**Name:** {self.name}\n\n"
        details += f"**Quantity:** {self.quantity}\n\n"
        details += f"**Price:** ${self.price if self.price else 'N/A'}\n\n"
        details += f"**Eco Score:** {self.eco_score}\n\n"
        details += f"**Eco Grade:** {self.eco_grade}\n\n"
        details += f"**Nutriscore:** {self.nutriscore}\n\n"
        details += f"**Nutriscore Grade:** {self.nutriscore_grade}\n\n"
        try:
            details += f"**Ingredients:** {', '.join(self.ingredients)}\n\n".replace(
                    "en:", ""
                ).replace("fr:", "")
        except Exception:
            details += "**Ingredients:** Not available\n\n"
        return details
