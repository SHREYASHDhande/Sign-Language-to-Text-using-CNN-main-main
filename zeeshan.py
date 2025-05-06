import numpy as np
import cv2
import os
import operator
import time
from string import ascii_uppercase
import tkinter as tk
from PIL import Image, ImageTk
from spellchecker import SpellChecker
from keras.models import model_from_json
import tensorflow as tf
import subprocess
# Set TensorFlow to use GPU if available
os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"
print(tf.__version__)

# Application Class
class Application:
    def __init__(self):

        # Initialize variables
        self.spell = SpellChecker()
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        # Load models
        self.load_models()

        # Setup UI
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)
        self.root.geometry("900x900")

        self.setup_ui()

        # Initialize variables
        self.str = ""  # Full sentence
        self.word = ""  # Current word
        self.current_symbol = "Empty"  # Last recognized symbol
        self.blank_flag = 0  # Flag to handle blank space
        self.ct = {char: 0 for char in ascii_uppercase}  # Character count
        self.ct['blank'] = 0

        self.video_loop()

    def load_models(self):
        # Loading all models
        self.model_paths = {
            "model_new": "Models/model_new.json",
            "model-bw_dru": "Models/model-bw_dru.json",
            "model-bw_tkdi": "Models/model-bw_tkdi.json",
            "model-bw_smn": "Models/model-bw_smn.json"
        }
        self.loaded_models = {}
        for model_name, model_path in self.model_paths.items():
            with open(model_path, "r") as json_file:
                model_json = json_file.read()
                model = model_from_json(model_json)
                weights_path = model_path.replace("json", "weights.h5")
                model.load_weights(weights_path)
                self.loaded_models[model_name] = model

        print("Models loaded successfully")

    def setup_ui(self):
        # UI Elements
        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=10, width=580, height=580)

        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=400, y=65, width=275, height=275)

        self.T = tk.Label(self.root, text="Sign Language To Text Conversion", font=("Courier", 30, "bold"))
        self.T.place(x=60, y=5)

        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=500, y=540)

        self.T1 = tk.Label(self.root, text="Character :", font=("Courier", 30, "bold"))
        self.T1.place(x=10, y=540)

        self.panel4 = tk.Label(self.root)
        self.panel4.place(x=220, y=595)

        self.T2 = tk.Label(self.root, text="Word :", font=("Courier", 30, "bold"))
        self.T2.place(x=10, y=595)

        self.panel5 = tk.Label(self.root)
        self.panel5.place(x=350, y=645)

        self.T3 = tk.Label(self.root, text="Sentence :", font=("Courier", 30, "bold"))
        self.T3.place(x=10, y=645)

        self.T4 = tk.Label(self.root, text="Suggestions :", fg="red", font=("Courier", 30, "bold"))
        self.T4.place(x=250, y=690)

        self.bt1 = tk.Button(self.root, command=self.action1, height=0, width=0)
        self.bt1.place(x=26, y=745)

        self.bt2 = tk.Button(self.root, command=self.action2, height=0, width=0)
        self.bt2.place(x=325, y=745)

        self.bt3 = tk.Button(self.root, command=self.action3, height=0, width=0)
        self.bt3.place(x=625, y=745)

        self.generate_button = tk.Button(
            self.root,
            text="Generate",
            command=self.generate_image,  # Updated to call generate_image method on button click
            font=("Courier", 20)
        )
        self.generate_button.place(x=650, y=650, width=150, height=50)

        # Add a Reset button
        self.reset_button = tk.Button(
            self.root,
            text="Reset",
            command=self.reset_text,
            font=("Courier", 20)
        )
        self.reset_button.place(x=450, y=650, width=150, height=50)

    def reset_text(self):
        """Reset the current word and sentence."""
        self.str = ""  # Clear the full sentence
        self.word = ""  # Clear the current word
        self.panel3.config(text="", font=("Courier", 30))  # Clear the symbol display
        self.panel4.config(text="", font=("Courier", 30))  # Clear the word display
        self.panel5.config(text="", font=("Courier", 30))  # Clear the sentence display
        print("Text reset successfully.")

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)

            # Define region of interest (ROI)
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            cv2image = cv2image[y1:y2, x1:x2]

            # Preprocess image for prediction
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            self.predict(res)

            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)

            # Update UI with current symbol, word, and sentence
            self.panel3.config(text=self.current_symbol, font=("Courier", 30))
            self.panel4.config(text=self.word, font=("Courier", 30))
            self.panel5.config(text=self.str, font=("Courier", 30))

            # Generate word suggestions
            predicts = sorted(self.spell.candidates(self.word))
            self.update_suggestions(predicts)

        self.root.after(5, self.video_loop)

    def generate_image(self):
        """Fetch the sentence and pass it as a prompt to generate an image."""
        # Fetch the sentence from the UI
        current_sentence = self.str.strip()  # Trim any extra whitespace
        
        # Ensure the sentence is not empty
        if not current_sentence:
            print("Error: No sentence available to generate an image.")
            return
        
        print(f"Generating image for prompt: {current_sentence}")

        # Call the image generation script with the sentence as input
        result = subprocess.run(
            ["python", "generate_image_with_prompt.py", current_sentence],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Display any errors in the console
        if result.returncode != 0:
            print("Error occurred during image generation:")
            print(result.stderr)
        else:
            print(result.stdout)
            self.show_generated_image("generated_image.jpg")  # Display the generated image

    def show_generated_image(self, image_path):
        """Display the generated image in a new window."""
        # Create a new top-level window
        new_window = tk.Toplevel(self.root)
        new_window.title("Generated Image")
        new_window.geometry("500x500")  # Set window size

        try:
            # Open the generated image
            img = Image.open(image_path)
            img = img.resize((400, 400))  # Resize the image to fit the new window
            imgtk = ImageTk.PhotoImage(img)

            # Add a label to display the image in the new window
            image_label = tk.Label(new_window, image=imgtk)
            image_label.imgtk = imgtk  # Keep a reference to prevent garbage collection
            image_label.pack(expand=True)  # Center the image in the new window

            print("Generated image displayed successfully!")
        except Exception as e:
            print(f"An error occurred while displaying the image: {e}")

    def predict(self, test_image):
        # Resize and normalize test image
        test_image = cv2.resize(test_image, (128, 128))
        test_image = test_image / 255.0  # Normalize for consistent model performance

        # Get prediction from the model
        result = self.loaded_models["model_new"].predict(test_image.reshape(1, 128, 128, 1))
        prediction = self.get_prediction(result)
        self.current_symbol = prediction[0][0]

        # Determine confidence of top prediction
        top_prediction_confidence = prediction[0][1]
        
        # Handle blank if confidence is too low
        confidence_threshold = 0.5  # Adjust this threshold based on model performance
        if top_prediction_confidence < confidence_threshold:
            self.current_symbol = "blank"

        # Update the character count and word formation logic
        self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > 60:
            for char in list(ascii_uppercase) + ['blank']:  # Fixed line
                self.ct[char] = 0  # Reset counts after confirmation of symbol

            if self.current_symbol == "blank":
                if not self.blank_flag:
                    self.blank_flag = 1
                    if len(self.word) > 0:
                        self.str += " " + self.word
                        self.word = ""
            else:
                self.blank_flag = 0
                self.word += self.current_symbol

    def get_prediction(self, result):
        prediction = {ascii_uppercase[i]: result[0][i+1] for i in range(26)}
        prediction["blank"] = result[0][0]  # Assuming "blank" is at index 0
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        return prediction

    def update_suggestions(self, predicts):
        if len(predicts) > 1:
            self.bt1.config(text=list(predicts)[0], font=("Courier", 20))
        else:
            self.bt1.config(text="")

        if len(predicts) > 2:
            self.bt2.config(text=list(predicts)[1], font=("Courier", 20))
        else:
            self.bt2.config(text="")

        if len(predicts) > 3:
            self.bt3.config(text=list(predicts)[2], font=("Courier", 20))
        else:
            self.bt3.config(text="")

    def action1(self):
        self.apply_suggestion(0)

    def action2(self):
        self.apply_suggestion(1)

    def action3(self):
        self.apply_suggestion(2)

    def apply_suggestion(self, index):
        predicts = list(self.spell.candidates(self.word))
        if index < len(predicts):
            self.word = ""
            self.str += " "
            self.str += predicts[index]

    def destructor(self):
        print("Closing application...")
        self.vs.release()
        cv2.destroyAllWindows()
        self.root.destroy()

# Run the application
print("Starting application...")
Application().root.mainloop()
