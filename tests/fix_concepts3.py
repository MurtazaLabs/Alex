with open(r'E:\Forex trading bot\dashboard\pages\4_Concepts\page.py', 'rb') as f:
    content = f.read()

# Fix the missing comma - find the exact pattern
# Looking for: b'            },\n"id": "concept_004"' (with indent 0 on the id line)
old = b'            },\n"id": "concept_004"'
new = b'            },\n            "id": "concept_004"'
if old in content:
    content = content.replace(old, new)
    print('Found and replaced!')
else:
    print('Pattern not found, trying alternative...')
    # Try alternative pattern
    if b'            },\n"id": "concept_004"' in content:
        content = content.replace(b'            },\n"id": "concept_004"', b'            },\n            "id": "concept_004"')
        print('Fixed with alt pattern!')
    else:
        print('Still not found')

with open(r'E:\Forex trading bot\dashboard\pages\4_Concepts\page.py', 'wb') as f:
    f.write(content)
print('Done!')