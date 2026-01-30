import pandas as pd

# Load the existing CSV
df = pd.read_csv('uploads/lotto1year.csv')

# New rows to add
new_rows = pd.DataFrame({
    'Draw ': [3464, 3465],
    'NLA': ['06-17-29-33-34 BB: 05 Letter: L', '01-04-13-18-27 BB: 24 Letter: O'],
    'Draw#': ['', ''],
    'SUP6': ['', '']
})

# Concatenate new rows at the top
df = pd.concat([new_rows, df], ignore_index=True)

# Save back to CSV
df.to_csv('uploads/lotto1year.csv', index=False)

print('New draws added to lotto1year.csv')
