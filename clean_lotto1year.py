import pandas as pd
import re

# Load the lotto1year.csv
df = pd.read_csv('uploads/lotto1year.csv')

parsed_data = []

for _, row in df.iterrows():
    nla = row['NLA']
    if pd.isna(nla) or not isinstance(nla, str):
        continue
    # Extract numbers from NLA string
    numbers = re.findall(r'\d+', nla)
    if len(numbers) < 5:
        continue
    nums = [int(n) for n in numbers[:5]]
    if not all(1 <= n <= 36 for n in nums):
        continue
    # Format as XX-XX-XX-XX-XX
    formatted_nla = '-'.join(f'{n:02d}' for n in nums)
    parsed_data.append({'NLA': formatted_nla})

# Create new DataFrame
new_df = pd.DataFrame(parsed_data)

# Save to desktop
new_df.to_csv('c:/Users/jomal/Desktop/cleaned_lotto1year.csv', index=False)

print(f"Cleaned CSV created with {len(new_df)} rows.")
