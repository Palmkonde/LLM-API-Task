import os
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

class GraphPloter:
    def __init__(self, api_key: str, model: str="mistral-large-latest") -> None:
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.running = True
    
    def ask_mistrial(self, user_input: str) -> None:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Extract the function name and interval from the following text. "
                    f"Respond in the format: 'function_name,x_min,x_max'. If the user wants to exit, respond with 'exit'.\n"
                    f"Input: {user_input}"
                )
            }
        ]

        chat_response = self.client.chat.complete(
            model=self.model,
            messages=messages
        )

        return chat_response.choices[0].message.content.strip()
    def plot_function(self, function_name: str, x_min: int | float, x_max: int | float) -> None:
        x = np.linspace(x_min, x_max, 500)
        if function_name == "sin":
            y = np.sin(x)
        elif function_name == "cos":
            y = np.cos(x)
        elif function_name == "x":
            y = x
        elif function_name == "x^2":
            y = x**2
        else:
            print("Unsupported function.")
            return

        plt.plot(x, y)
        plt.title(f"{function_name} on [{x_min}, {x_max}]")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        plt.show()
    
    def run_main(self) -> None:
        while self.running:
            user_input = input("Enter your request: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                self.running = False
                print("Goodbye!")
                continue

            response = self.ask_mistral(user_input)
            if response.lower() == "exit":
                self.running = False
                print("Goodbye!")
            else:
                try:
                    function_name, x_min, x_max = response.split(",")
                    self.plot_function(function_name, float(x_min), float(x_max))
                except ValueError:
                    print("Sorry, I couldn't understand your input. Please try again.")

if __name__ == "__main__":
    api_key = os.environ["MISTRAL_API_KEY"]
    plotter = GraphPloter(api_key=api_key)
    plotter.run_main()
