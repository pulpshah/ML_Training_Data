# Sentence Type Classification
Objective: The goal is to develop a model that can classify a given sentence as exclamatory, imperative, declarative, or interrogative.

Dataset: For training and evaluation, we will use transcripts from presidential debates. Each sentence in the dataset will be manually labeled with one of the following types:

Exclamatory: Sentences that express strong emotion, usually ending with an exclamation mark.

Imperative: Sentences that give commands, requests, or instructions.

Declarative: Sentences that make statements or provide information.

Interrogative: Sentences that ask questions, usually ending with a question mark.

Model Choice: We will use a transformer-based model, as these models have shown great success in understanding and processing natural language. Transformers are well-suited for this task because they can capture the context and nuances of sentences, allowing for accurate classification.

Steps:
Data Collection: Gather transcripts from various presidential debates and Generate sentences using GPT-4.

Data Labeling: label each sentence as exclamatory, imperative, declarative, or interrogative.

Model Selection: Choose a transformer model, such as BERT or RoBERTa, for the classification task.

Training: Train the model on the labeled dataset.

Evaluation: Evaluate the model’s performance using appropriate metrics (accuracy, precision, recall, F1 score).

Deployment: Deploy the model to a suitable environment for practical use.

Number of sentences: 5000

## Model Use:

``` Python
from transformers import BertTokenizer # Load BertForArgumentScoring instead of BertForSequenceClassification

tokenizer = RobertaTokenizer.from_pretrained('/content/argument_scoring_model')
model = RobertaForArgumentScoring.from_pretrained('/content/argument_scoring_model') # Load BertForArgumentScoring instead of BertForSequenceClassification
# Set the model to evaluation mode
model.eval()

def prepare_input(text):
    return tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

# Example input text
text = "magine a mother who has to choose between paying for her child’s life-saving medication or keeping the lights on at home. Every day, countless families face this agonizing decision because the cost of healthcare in the U.S. is so high. This isn’t just a statistic—it’s a heartbreaking reality that affects real people. The emotional toll of watching loved ones suffer due to unaffordable care is immense and unacceptable. We need to act now to ensure that every individual, regardless of their financial situation, has access to the healthcare they need. It’s time to put compassion into action and make healthcare affordable for everyone."

# Prepare the input
inputs = prepare_input(text)

# Run the input through the model
# Move the input tensors to the same device as the model
with torch.no_grad():
    outputs = model(
        input_ids=inputs['input_ids'].to(model.device), # Move input_ids to the model's device
        attention_mask=inputs['attention_mask'].to(model.device) # Move attention_mask to the model's device
    )

# Extract scores from the output
scores = outputs['logits'].squeeze().tolist()

# Print the results
print(f"Logos Score: {scores[0]:.2f}")
print(f"Pathos Score: {scores[1]:.2f}")
print(f"Ethos Score: {scores[2]:.2f}")

```
# Sentence Structure Classification
Objective: The goal is to develop a model that can classify a given sentence as simple, compound, or compound-complex.

Dataset: For training and evaluation, we will use transcripts from presidential debates. Each sentence in the dataset will be manually labeled with one of the following types:
Simple: Sentences that contain a single independent clause.
Compound: Sentences that contain two or more independent clauses, usually joined by a conjunction.
Compound-Complex: Sentences that contain at least two independent clauses and one or more dependent clauses.

Model Choice: We will use a transformer-based model for this task as well. Transformers are capable of understanding the syntactic structure of sentences, making them suitable for distinguishing between simple, compound, and compound-complex sentences.

Steps:

Data Collection: Gather transcripts from various presidential debates and Generate sentences using GPT-4.

Data Labeling: label each sentence as simple, compound, or compound-complex.

Model Selection: Choose a transformer model, such as BERT or RoBERTa, for the classification task.

Training: Train the model on the labeled dataset.

Evaluation: Evaluate the model’s performance using appropriate metrics (accuracy, precision, recall, F1 score).

Deployment: Deploy the model to a suitable environment for practical use.

Number of sentences: 5000


