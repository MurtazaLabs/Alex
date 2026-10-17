with open(r'E:\Forex trading bot\dashboard\pages\4_Concepts\page.py', 'rb') as f:
    content = f.read()
content = content.replace(b'            },\n            "id": "concept_004"', b'            },\n            "id": "concept_004"')
with open(r'E:\Forex trading bot\dashboard\pages\4_Concepts\page.py', 'wb') as f:
    f.write(content)
print('Fixed!')