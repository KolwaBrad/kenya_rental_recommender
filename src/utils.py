import pandas as pd
import re
from markupsafe import Markup

def format_price(price):
    return f"KSh {price:,.0f}"

def prepare_results(recommended_properties):
    for prop in recommended_properties:
        prop['Price'] = format_price(prop['Price'])
    return recommended_properties


def prepare_results(properties_df):
    """
    Prepare the properties DataFrame for display
    """
    if isinstance(properties_df, pd.DataFrame):
        return properties_df.to_dict('records')
    return properties_df

def format_gemini_response(response_text):
    """
    Format the Gemini API response text to proper HTML/Markdown format.
    
    Args:
        response_text (str): Raw text response from Gemini API
    
    Returns:
        str: Formatted HTML text
    """
    # Replace double asterisks with strong tags for bold text
    formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', response_text)
    
    # Split text by double newlines to identify paragraphs
    paragraphs = formatted_text.split('\n\n')
    
    # Wrap each paragraph in <p> tags
    formatted_paragraphs = ['<p class="mb-4">{}</p>'.format(p.strip()) for p in paragraphs if p.strip()]
    
    # Join paragraphs with newlines for proper spacing
    final_text = '\n'.join(formatted_paragraphs)
    
    # Make the text safe for HTML rendering
    return Markup(final_text)

def clean_property_description(description):
    """
    Clean and structure property descriptions by removing asterisks and adding proper HTML structure.
    
    Args:
        description (str): Raw property description
    
    Returns:
        str: Cleaned and formatted HTML description
    """
    # Remove single asterisks that aren't part of pairs
    description = re.sub(r'(?<!\*)\*(?!\*)', '', description)
    
    # Split into sections (Amenities, Recent News, Property Features)
    sections = re.split(r'\*\*(.*?)\*\*:', description)
    
    formatted_sections = []
    for i in range(1, len(sections), 2):
        if i + 1 < len(sections):
            section_title = sections[i].strip()
            section_content = sections[i + 1].strip()
            formatted_sections.append(
                f'<div class="mb-6">'
                f'<h3 class="text-lg font-semibold mb-2">{section_title}</h3>'
                f'<p class="text-gray-700">{section_content}</p>'
                f'</div>'
            )
    
    return Markup(''.join(formatted_sections))