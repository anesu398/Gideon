import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# Load pre-trained GPT2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# Define your training data (sample data for demonstration)
training_data = [
    "Physics is the study of matter, energy, and the fundamental forces of nature.",
    "In electronics, circuits are used to control the flow of electric current.",
    "Computer science involves the study of algorithms, data structures, and programming languages."
]

# Tokenize and encode the training data
input_ids = tokenizer(training_data, return_tensors="tf", padding=True, truncation=True)["input_ids"]

# Define model training parameters
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Train the model
for epoch in range(3):  # Example: Train for 3 epochs
    for batch in input_ids:
        with tf.GradientTape() as tape:
            outputs = model(batch)
            logits = outputs.logits[:, :-1, :]  # Remove last token for input
            labels = batch[:, 1:]  # Shift labels to the right
            mask = tf.math.logical_not(tf.math.equal(labels, tokenizer.pad_token_id))  # Create mask to ignore padding
            loss_value = loss(labels, logits, sample_weight=tf.cast(mask, tf.float32))
        
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    print(f"Epoch {epoch+1}: Loss = {loss_value.numpy()}")

# Save the trained model
model.save_pretrained("gideon_model")
tokenizer.save_pretrained("gideon_tokenizer")

# Load the trained model and tokenizer
loaded_tokenizer = GPT2Tokenizer.from_pretrained("gideon_tokenizer")
loaded_model = TFGPT2LMHeadModel.from_pretrained("gideon_model", pad_token_id=loaded_tokenizer.eos_token_id)

# Example usage: Generate text
prompt = "The laws of physics"
input_ids = loaded_tokenizer(prompt, return_tensors="tf")["input_ids"]
output_ids = loaded_model.generate(input_ids, max_length=100, num_return_sequences=1)
generated_text = loaded_tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated text:", generated_text)
