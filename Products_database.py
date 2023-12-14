from Products import Product
import numpy as np


class ProductDatabase:
    """
    A class to handle operations on the product database.
    """

    def __init__(self, json_data):
        """
        Initialize the ProductDatabase with data from the JSON file.
        """
        self.products = {
            product_id: Product(data) for product_id, data in json_data.items()
        }

    def search_products(self, query, limit=20):
        # Search products using edit distance
        scored_products = [
            (levenshtein_distance(query.lower(), prod.name.lower()), prod)
            for prod in self.products.values()
            if prod and prod.name
        ]
        scored_products.sort(key=lambda x: x[0])  # Sort by edit distance
        return [prod for _, prod in scored_products][:limit]

    def get_product(self, product_id):
        """
        Retrieve a single product by its ID.
        """
        return self.products.get(product_id)


def levenshtein_distance(s1, s2):
    """
    Compute the Levenshtein distance between two strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    # Initialize the distance matrix
    distances = np.zeros((len(s2) + 1, len(s1) + 1))

    for t1 in range(len(s1) + 1):
        distances[0][t1] = t1

    for t2 in range(len(s2) + 1):
        distances[t2][0] = t2

    # Populate the distance matrix
    for t2 in range(1, len(s2) + 1):
        for t1 in range(1, len(s1) + 1):
            cost = 0 if s1[t1 - 1] == s2[t2 - 1] else 1
            distances[t2][t1] = min(
                distances[t2 - 1][t1] + 1,  # deletion
                distances[t2][t1 - 1] + 1,  # insertion
                distances[t2 - 1][t1 - 1] + cost,
            )  # substitution

    return distances[t2][t1]
