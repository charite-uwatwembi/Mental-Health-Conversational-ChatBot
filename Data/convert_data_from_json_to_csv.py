import json
import pandas as pd

# Load JSON data
with open('Data/intents.json', 'r') as f:
    data = json.load(f)

# Create training CSV (patterns + tags + responses)
training_data = []
seen_questions = set()  # Track unique questions

for intent in data['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        if pattern not in seen_questions:  # Avoid duplicates
            seen_questions.add(pattern)
            response = intent['responses'][0]  # Pick the first response
            training_data.append({
                'question': pattern,  
                'answer': response,  
                'pattern': pattern,  
                'tag': tag  
            })

# Save to CSV
df = pd.DataFrame(training_data)
df.to_csv('Data/mental_health_training.csv', index=False)

print("CSV file has been created successfully without duplicate questions.")