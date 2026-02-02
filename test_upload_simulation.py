import pandas as pd
import re
import os

# Simulate the upload process for test_draw_2551_full.csv
def simulate_upload():
    filepath = 'test_draw_2551_full.csv'
    if not os.path.exists(filepath):
        print("Test CSV not found")
        return

    # Load and validate data
    df = pd.read_csv(filepath)
    if 'NLA' not in df.columns:
        print("Error: CSV must have a column named 'NLA'.")
        return
    if len(df) < 1:
        print("Error: CSV must have at least 1 row.")
        return

    # Parse NLA column to extract 5 numbers, BB, and Letter per row
    parsed_data = []
    for _, row in df.iterrows():
        nla = row['NLA']
        if pd.isna(nla) or not isinstance(nla, str):
            continue
        # Extract numbers from NLA string
        import re
        numbers = re.findall(r'\d+', nla)
        if len(numbers) < 5:
            print(f"Error: Row '{nla}' does not contain at least 5 numbers.")
            return
        nums = [int(n) for n in numbers[:5]]
        if not all(1 <= n <= 36 for n in nums):
            print(f"Error: Numbers in '{nla}' must be between 1 and 36.")
            return
        # Extract BB and Letter, add defaults if missing
        bb_match = re.search(r'BB[:\-]\s*(\d+)', nla)
        letter_match = re.search(r'Letter:\s*([A-Z])', nla)
        bb = int(bb_match.group(1)) if bb_match else 1
        letter = letter_match.group(1) if letter_match else 'A'

        # Parse SUP6 if present
        sup_parsed = [0] * 8  # 6 nums + bb + letter
        if 'SUP6' in df.columns and not pd.isna(row['SUP6']) and isinstance(row['SUP6'], str):
            sup6 = row['SUP6']
            numbers_sup = re.findall(r'\d+', sup6)
            if len(numbers_sup) >= 6:
                nums_sup = [int(n) for n in numbers_sup[:6]]
                if all(1 <= n <= 36 for n in nums_sup):
                    bb_match_sup = re.search(r'BB[:\-]\s*(\d+)', sup6)
                    letter_match_sup = re.search(r'Letter:\s*([A-Z])', sup6)
                    bb_sup = int(bb_match_sup.group(1)) if bb_match_sup else 1
                    letter_sup = letter_match_sup.group(1) if letter_match_sup else 'A'
                    # Check for letter at the end if not found in regex
                    if letter_match_sup is None:
                        parts = sup6.split()
                        for part in reversed(parts):
                            if part.isalpha() and len(part) == 1:
                                letter_sup = part.upper()
                                break
                    sup_parsed = nums_sup + [bb_sup, letter_sup]

        # Include Draw and Draw# if available
        draw_val = row.get('Draw ', None)
        draw_num_val = row.get('Draw#', None)
        parsed_data.append([draw_val, draw_num_val] + nums + [bb, letter] + sup_parsed)

    df_parsed = pd.DataFrame(parsed_data, columns=['Draw', 'Draw#', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'BB', 'Letter', 'Sup1', 'Sup2', 'Sup3', 'Sup4', 'Sup5', 'Sup6', 'SupBB', 'SupLetter'])

    # Save to reconstructed_lotto.csv
    df_parsed.to_csv('reconstructed_lotto_test.csv', index=False)

    print("Parsed data:")
    print(df_parsed)
    print("\nSaved to reconstructed_lotto_test.csv")

    # Check if Draw# 2551 is present and SUP6 letter is 'J'
    row_2551 = df_parsed[df_parsed['Draw#'] == 2551]
    if not row_2551.empty:
        sup_letter = row_2551['SupLetter'].values[0]
        print(f"Draw# 2551 SUP6 Letter: {sup_letter}")
        if sup_letter == 'J':
            print("SUCCESS: Draw# 2551 SUP6 letter saved correctly as 'J'")
        else:
            print(f"FAILURE: Expected 'J', got '{sup_letter}'")
    else:
        print("FAILURE: Draw# 2551 not found in parsed data")

if __name__ == '__main__':
    simulate_upload()
