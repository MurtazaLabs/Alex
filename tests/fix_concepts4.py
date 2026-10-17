with open(r'E:\Forex trading bot\dashboard\pages\4_Concepts\page.py', 'rb') as f:
    content = f.read()

# Fix: add missing { before concept_004
old = b'            },\n            "id": "concept_004"'
new = b'            },\n            {\n            "id": "concept_004"'
if old in content:
    content = content.replace(old, new)
    print('Fixed!')
else:
    print('Pattern not found')

with open(r'E:\Forex trading bot\dashboard\pages\4_Concepts\page.py', 'wb') as f:
    f.write(content)
print('Done!')