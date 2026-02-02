import pandas as pd
import re

# Test the SUP6 parsing logic from the index route
def test_sup6_parsing():
    # Simulate a row with SUP6 "02 12 14 15 16 18 J"
    row = {'SUP6': '02 12 14 15 16 18 J', 'Draw#': 2551}

    sup_parsed = [0] * 8  # 6 nums + bb + letter
    if 'SUP6' in row and not pd.isna(row['SUP6']) and isinstance(row['SUP6'], str):
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

    print(f"Parsed SUP6: {sup_parsed}")
    print(f"Numbers: {sup_parsed[:6]}, BB: {sup_parsed[6]}, Letter: {sup_parsed[7]}")

    # Check if letter is 'J'
    assert sup_parsed[7] == 'J', f"Expected 'J', got '{sup_parsed[7]}'"
    print("Test passed: Letter parsed correctly as 'J'")

if __name__ == '__main__':
    test_sup6_parsing()
