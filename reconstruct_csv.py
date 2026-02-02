import csv

# Read the cleaned_lotto_copy2.csv
with open('cleaned_lotto_copy2.csv', 'r') as infile:
    reader = csv.DictReader(infile)
    rows = list(reader)

# Prepare the new rows
new_rows = []
for row in rows:
    # Construct NLA
    nla = f"{row['Num1']} {row['Num2']} {row['Num3']} {row['Num4']} {row['Num5']} BB: {row['BB']} Letter: {row['Letter']}"
    # Construct SUP6
    sup6 = f"{row['Sup1']} {row['Sup2']} {row['Sup3']} {row['Sup4']} {row['Sup5']} {row['Sup6']} BB: {row['SupBB']} Letter: {row['SupLetter']}"
    # Draw and Draw#
    draw = row['Draw']
    draw_num = row['Draw#']
    new_rows.append({
        'NLA': nla,
        'SUP6': sup6,
        'Draw ': draw,
        'Draw#': draw_num
    })

# Write to new CSV
with open('reconstructed_lotto.csv', 'w', newline='') as outfile:
    fieldnames = ['NLA', 'SUP6', 'Draw ', 'Draw#']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in new_rows:
        writer.writerow(row)

print("Reconstructed CSV created as reconstructed_lotto.csv")
