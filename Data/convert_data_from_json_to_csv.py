import json
import pandas as pd

# Load JSON data
with open('intents.json', 'r') as f:
    data = json.load(f)

# Create training CSV (patterns + tags + responses)
training_data = []
seen_questions = set()

for intent in data['intents']:
    tag = intent['tags'][0]  # Corrected here
    for pattern in intent['patterns']:
        if pattern not in seen_questions:
            seen_questions.add(pattern)
            response = intent['responses'][0]
            training_data.append({
                'question': pattern,  
                'answer': response,  
                'pattern': pattern,  
                'tag': tag  
            })

# Save to CSV
df = pd.DataFrame(training_data)
df.to_csv('mental_health_training.csv', index=False)

print("CSV file has been created successfully without duplicate questions.")