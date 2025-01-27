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
                    "Extract the function name and interval from the following text. "
                    "Respond in the format: 'function_name, [coeffician1, coeffician2, coeiffician3,... ], x_min, x_max'. If the user wants to exit, respond with 'exit'.\n"
                    "For example: user_input: 3x^3 + 2x^2 + 2x + 1 (-10, 10) respond: poly, [3, 2, 2, 1], -10, 10"
                    "Another example: user_input: sin(2x) (-2, 5) respond: sin, [2], -2, 5"
                    f"Input: {user_input}"
                )
            }
        ]

        chat_response = self.client.chat.complete(
            model=self.model,
            messages=messages
        )

        return chat_response.choices[0].message.content.strip()
    def plot_function(self, function_name: str, coefficients: list[int | float], x_min: int | float, x_max: int | float) -> None:
        x = np.linspace(x_min, x_max, 500)

        if function_name == "sin":
            k = coefficients[0]  
            y = np.sin(k * x)
        elif function_name == "cos":
            k = coefficients[0]
            y = np.cos(k * x)
        elif function_name == "poly":
            y = np.polyval(coefficients, x)
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

            response = self.ask_mistrial(user_input)
            if response.lower() == "exit":
                self.running = False
                print("Goodbye!")
            else:
                try:
                    parts = response.split(",")
                    function_name = parts[0].strip()

                    flag = False 
                    coefficients_str = ""
                    nxt = -1
                    for i, e in enumerate(parts):
                        if "[" in e and "]" in e:
                            coefficients_str = e.strip()
                            nxt = i + 1
                            break

                        if "[" in parts[i]:
                            coefficients_str += e.lstrip() + ','
                            flag = True
                        
                        elif "]" in parts[i]:
                            coefficients_str += e.strip()
                            nxt = i + 1
                            break

                        elif flag:
                            coefficients_str += e.rstrip() + ','
                        
                    coefficients = eval(coefficients_str) 
                    # coefficients = coefficients[0]
                    
                    x_min = float(parts[nxt].strip())
                    x_max = float(parts[nxt + 1].strip())

                    self.plot_function(function_name, coefficients, x_min, x_max)

                except ValueError as e:
                    print(f"Sorry, I couldn't understand your input. Please try again. Error: {e}")
                except IndexError:
                    print("The input format is incorrect. Please follow the format 'function_name, [coefficients], x_min, x_max'.")
                except SyntaxError:
                    print("There seems to be a syntax issue with the coefficients. Make sure they are in list format.")

if __name__ == "__main__":
    api_key = os.environ["MISTRAL_API_KEY"]
    plotter = GraphPloter(api_key=api_key)
    plotter.run_main()
