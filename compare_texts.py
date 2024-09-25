from difflib import ndiff, SequenceMatcher
import re

# Correct text
correct_text = """Uh H-U-G-H. Thanks for entrusting Annabelle, your 18-kilogram patient, who’s up in weight and who received doxorubicin today as adjuvant therapy post-splenectomy for the treatment of hemangiosarcoma. As you recall, the dog had a hemoabdomen when the surgery was done to remove the spleen that resulted in the diagnosis. This represents the third out of five doxorubicin chemotherapy treatments. Staging may be wise prior to the next treatment."""

# Erroneous text
erroneous_text = """Annabelle, H-U-G-H, thanks for listening. Annabelle, you're an 18-kilogram patient who's up in weight and who received Dr. Rubison today's adjuvant therapy post-spleenectomy for the treatment of Mangia sarcoma. As you recall, the dog had a hemoabdomen when the surgery was done to remove the spleen that resulted in the diagnosis. This represents a third out of five doctors doing therapy treatments. Staging may be wise prior to the next treatment."""

# Function to highlight differences
def highlight_differences(correct_text, erroneous_text):
    diff = list(ndiff(correct_text.split(), erroneous_text.split()))
    highlighted = []
    
    for word in diff:
        if word.startswith("-"):
            highlighted.append(f"<span style='color:red'>{word[2:]}</span>")
        elif not word.startswith("+"):
            highlighted.append(word[2:])
    
    return " ".join(highlighted)

# Function to calculate WER
def calculate_wer(correct_text, erroneous_text):
    correct_words = correct_text.split()
    erroneous_words = erroneous_text.split()
    
    sm = SequenceMatcher(None, correct_words, erroneous_words)
    distance = len(correct_words) + len(erroneous_words) - 2 * sum(n for i, j, n in sm.get_matching_blocks())
    
    return (distance / len(correct_words)) * 100

# Highlight differences in the text
highlighted_text = highlight_differences(correct_text, erroneous_text)

# Calculate WER
wer = calculate_wer(correct_text, erroneous_text)

# Display results
from IPython.display import display, HTML

display(HTML(f"<p>{highlighted_text}</p>"))
print(f"Word Error Rate (WER): {wer:.2f}%")
