import pandas as pd
import re

# Load the lotto1year.csv
df = pd.read_csv('uploads/lotto1year.csv')

def process_nla(text):
    if pd.isna(text) or not isinstance(text, str):
        return ''
    # Find numbers
    numbers = re.findall(r'\d+', text)
    if len(numbers) < 5:
        return ''
    nums = [int(n) for n in numbers[:5]]
    if not all(1 <= n <= 36 for n in nums):
        return ''
    nla_str = '-'.join(f'{n:02d}' for n in nums)
    # Check for BB
    bb_match = re.search(r'BB[:\-]\s*(\d+)', text)
    bb = bb_match.group(1) if bb_match else '01'
    # Check for Letter
    letter_match = re.search(r'Letter:\s*([A-Z])', text)
    if not letter_match:
        # Find single letter not part of words
        letters = re.findall(r'\b([A-Z])\b', text)
        letter = letters[0] if letters else 'A'
    else:
        letter = letter_match.group(1)
    return f'{nla_str} BB: {bb} Letter: {letter}'

df['NLA'] = df['NLA'].apply(process_nla)

# Save to desktop
df.to_csv('c:/Users/jomal/Desktop/processed_lotto1year.csv', index=False)

print(f"Processed CSV created with {len(df)} rows.")
