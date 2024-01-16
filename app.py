# Import necessary modules
import sys
from transformers import BertTokenizer, BertForTokenClassification
from torch.nn.functional import softmax

# Load Pre-Trained Tokenizer and Model:
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
model = BertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')

# Define a Function to Perform NER and Aggregate Results:
def perform_ner(text):
    # Encode and tokenize the input text
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    inputs = tokenizer.encode(text, return_tensors="pt")

    # Get model predictions
    outputs = model(inputs).logits
    predictions = softmax(outputs, dim=2)

    # Process and aggregate entity predictions
    entities = []
    current_entity = ""
    current_entity_type = ""
    for token, prediction in zip(tokens, predictions[0].argmax(dim=-1)):
        entity = model.config.id2label[prediction.item()]

        # Check if the token is part of an entity
        if entity != 'O':
            # Remove 'B-' or 'I-' prefix from entity type
            entity_type = entity[2:]

            # Aggregate tokens of the same entity
            if current_entity_type == entity_type:
                current_entity += " " + token if token.startswith("##") else token
            else:
                if current_entity:
                    entities.append((current_entity, current_entity_type))
                current_entity = token
                current_entity_type = entity_type
        else:
            if current_entity:
                entities.append((current_entity, current_entity_type))
                current_entity = ""
                current_entity_type = ""

    # Add last entity if any
    if current_entity:
        entities.append((current_entity, current_entity_type))

    # Post-process tokens to remove '##'
    entities = [(entity.replace('##', ''), etype) for entity, etype in entities]
    return entities

# # Analyze Text for Named Entities:
# text = "Hugging Face is based in New York City."
# entities = perform_ner(text)
# print(f"Named Entities: {entities}")

def main():
    print("Hugging Face NER Chatbot")
    print("Type 'exit' to end the conversation.\n")

    while True:
        try:
            # Input from user
            text = input("You: ")
            
            # Check for exit command
            if text.lower() == 'exit':
                print("Exiting the chatbot.")
                break

            # Perform NER on the input text
            entities = perform_ner(text)
            if entities:
                print("Named Entities:")
                for entity, entity_type in entities:
                    print(f" - {entity} ({entity_type})")
            else:
                print("No named entities found.")

        except KeyboardInterrupt:
            # Handling Ctrl+C interruption
            print("\nExiting the chatbot.")
            break
        except Exception as e:
            # Handling other exceptions
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()
