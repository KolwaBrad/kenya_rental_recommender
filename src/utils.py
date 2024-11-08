import pandas as pd

def format_price(price):
    return f"KSh {price:,.0f}"

def prepare_results(recommended_properties):
    for prop in recommended_properties:
        prop['Price'] = format_price(prop['Price'])
    return recommended_properties