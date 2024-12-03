import tkinter as tk
from tkinter import messagebox
import pickle

# Load the trained model and any necessary preprocessing tool (e.g., vectorizer)
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Predict whether an email is spam or ham
def classify_email():
    email_text = email_input.get("1.0", tk.END).strip()
    if not email_text:
        messagebox.showwarning("Input Error", "Please enter an email text!")
        return
    
    try:
        # Preprocess and predict
        processed_text = vectorizer.transform([email_text])
        prediction = model.predict(processed_text)[0]
        
        # Update the result label
        result_text = "Spam" if prediction == 1 else "Ham"
        result_label.config(text=f"Result: {result_text}", fg="green" if prediction == 0 else "red")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during classification: {str(e)}")

# Initialize the main window
root = tk.Tk()
root.title("Email Spam Classifier")
root.geometry("500x400")

# Load model and vectorizer
try:
    model, vectorizer = load_model_and_vectorizer()
except Exception as e:
    messagebox.showerror("Initialization Error", f"Failed to load model/vectorizer: {str(e)}")
    root.destroy()

# Create UI components
title_label = tk.Label(root, text="Email Spam Classifier", font=("Helvetica", 18, "bold"))
title_label.pack(pady=10)

instructions_label = tk.Label(root, text="Enter the email text below:", font=("Helvetica", 12))
instructions_label.pack()

email_input = tk.Text(root, wrap=tk.WORD, height=10, width=50, font=("Helvetica", 12))
email_input.pack(pady=10)

classify_button = tk.Button(root, text="Classify", command=classify_email, font=("Helvetica", 12))
classify_button.pack(pady=10)

result_label = tk.Label(root, text="Result: None", font=("Helvetica", 14))
result_label.pack(pady=20)

# Start the main event loop
root.mainloop()
