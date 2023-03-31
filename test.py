from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
import tensorflow as tf

model_path ='analysis-model'

# Load the saved model from the local directory
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)


# Load the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Example text
text = "I haven't experienced a terrible occurrence.I have just landed his dream job, but he feels a great deal of pressure to perform well. He approaches his work with a positive mentality, recognizing that he has the skills and abilities to succeed. Mark takes breaks throughout the day to recharge his batteries and make time for his personal interests and hobbies. He prioritizes getting a good night's sleep and follows a regular sleep schedule, which helps him to stay focused and alert during the day. Mark makes healthy eating choices and ensures that he has balanced meals throughout the day. He has a supportive network of friends and family who are proud of his accomplishments and provide him with unwavering support. Mark feels comfortable sharing his fears and concerns with them, knowing that they will not judge him. This supportive network helps him to stay positive and confident, and he is excelling in his new role. Mark is grateful for the love and support he receives and feels confident in his ability to succeed in his new job."

# Tokenize the text
inputs = tokenizer(text, truncation=True, padding=True, return_tensors='tf')

# Make predictions
outputs = model(inputs)
predictions = tf.argmax(outputs.logits, axis=1).numpy()

print(predictions)