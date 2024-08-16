# Rhetorical Appeal Scoring

**Objective:**  
The goal is to develop a model that can take in a sentence and provide a score between 0 and 1 for the rhetorical appeals of pathos, logos, and ethos. These scores will help in understanding the persuasive strategies used in the text.

**Dataset:**  
For training and evaluation, we will use a dataset consisting of 6000 sentences extracted from various sources, such as Reddit comments, speeches, or essays. Each sentence in the dataset will be manually labeled with scores for the following rhetorical appeals:

- **Pathos:** The appeal to emotion, reflecting how the sentence evokes feelings.
- **Logos:** The appeal to logic, indicating the sentence's use of reasoning and evidence.
- **Ethos:** The appeal to credibility, representing how the sentence conveys authority and trustworthiness.

**Model Choice:**  
We will use a transformer-based model, such as RoBERTa, for scoring the rhetorical appeals. Transformers are well-suited for this task because they can capture the contextual meaning and nuances of sentences, enabling accurate scoring across the three appeals.

**Steps:**

1. **Data Collection:**
   - Gather sentences from Reddit r\changemyview using a Reddit scraper.

2. **Data Labeling:**
   - Use GPT-4 to score each text on patho, logos, and ethos.
  
3. **Model Selection:**
   - Choose a pre-trained transformer model, such as RoBERTa, as the base for the scoring task.
   - Fine-tune the model on the labeled dataset to adjust it to the specific task of scoring rhetorical appeals.

4. **Training:**
   - Train the model on the labeled dataset using appropriate loss functions for regression tasks (e.g., Mean Squared Error).
   - Implement techniques like cross-validation to ensure the model's robustness.

5. **Evaluation:**
   - Evaluate the model's performance using metrics such as Mean Absolute Error (MAE) and R-squared to measure how well the predicted scores align with the true labels.
   - Conduct qualitative analysis by comparing the model's output to human judgments.

6. **Deployment:**
   - Deploy the trained model to a suitable environment, such as a web application or API, where it can be used to analyze new sentences for their rhetorical appeal scores.
   - Monitor the model's performance and update it as necessary based on user feedback or additional data.

**Number of Sentences:**  
6000


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

**Output:**

```
Logos Score: 0.56
Pathos Score: 0.93
Ethos Score: 0.34
``` 
