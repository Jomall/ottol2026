import pandas as pd
import re

# Load the original CSV
df = pd.read_csv('uploads/lotto1year.csv')

# Filter rows where NLA is not NaN and is a string
valid_rows = []
for _, row in df.iterrows():
    nla = row['NLA']
    if pd.isna(nla) or not isinstance(nla, str):
        continue
    # Try to extract 5 numbers
    numbers = re.findall(r'\d+', nla)
    if len(numbers) >= 5:
        # Reconstruct a clean NLA format
        nums = numbers[:5]
        bb = numbers[5] if len(numbers) > 5 else '29'  # Default BB
        letter = 'K'  # Default letter
        clean_nla = f"{nums[0]}-{nums[1]}-{nums[2]}-{nums[3]}-{nums[4]} BB: {bb} Letter: {letter}"
        valid_rows.append(clean_nla)

# Create a new DataFrame with NLA column
new_df = pd.DataFrame({'NLA': valid_rows})

# Save to new CSV
new_df.to_csv('cleaned_lotto.csv', index=False)

print(f"Created cleaned_lotto.csv with {len(valid_rows)} rows.")
