import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

# 1. User says: “I have chicken, rice, egg, and onion.”
# 2. Model sees the tools list and decides suggest_recipe is useful.
# 3. Model returns a tool call, like: call suggest_recipe with ingredients.
# 4. Your Python code parses those arguments.
# 5. Your code runs suggest_recipe(...).
# 6. The result is sent back to the model.
# 7. The model turns the raw recipe result into a friendly answer.


load_dotenv()


def create_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not configured")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )


def normalize_ingredients(ingredients):
    if not isinstance(ingredients, list):
        raise ValueError("ingredients must be a list")

    normalized = []

    for ingredient in ingredients:
        if not isinstance(ingredient, str) or not ingredient.strip():
            raise ValueError("each ingredient must be a non-empty string")

        normalized.append(ingredient.strip().lower())

    return normalized


def suggest_recipe(ingredients):
    normalized_ingredients = normalize_ingredients(ingredients)
    available_ingredients = set(normalized_ingredients)

    recipes = [
        {
            "name": "Chicken Fried Rice",
            "ingredients": ["chicken", "rice", "egg", "onion"],
            "time": 25,
            "steps": [
                "Cook the rice.",
                "Fry chicken and onion.",
                "Add egg and rice.",
                "Season and stir-fry."
            ]
        },
        {
            "name": "Tomato Egg Noodles",
            "ingredients": ["tomato", "egg", "noodles"],
            "time": 15,
            "steps": [
                "Boil the noodles.",
                "Cook tomato in a pan.",
                "Add scrambled eggs.",
                "Mix everything together."
            ]
        }
    ]

    results = []

    for recipe in recipes:
        required_ingredients = set(recipe["ingredients"])

        if required_ingredients.issubset(available_ingredients):
            results.append({
                "name": recipe["name"],
                "time": recipe["time"],
                "matched_ingredients": recipe["ingredients"],
                "steps": recipe["steps"]
            })

    return results


def estimate_nutrition(ingredients):
    """Estimate nutrition using a typical serving size for each ingredient."""
    normalized_ingredients = normalize_ingredients(ingredients)

    nutrition_data = {
        "chicken": {
            "serving": "100 g cooked chicken breast",
            "calories": 165,
            "protein_g": 31,
            "carbs_g": 0,
            "fat_g": 3.6
        },
        "rice": {
            "serving": "1 cup cooked rice",
            "calories": 205,
            "protein_g": 4.3,
            "carbs_g": 44.5,
            "fat_g": 0.4
        },
        "egg": {
            "serving": "1 large egg",
            "calories": 72,
            "protein_g": 6.3,
            "carbs_g": 0.4,
            "fat_g": 4.8
        },
        "onion": {
            "serving": "1 medium onion",
            "calories": 44,
            "protein_g": 1.2,
            "carbs_g": 10.3,
            "fat_g": 0.1
        },
        "tomato": {
            "serving": "1 medium tomato",
            "calories": 22,
            "protein_g": 1.1,
            "carbs_g": 4.8,
            "fat_g": 0.2
        },
        "noodles": {
            "serving": "1 cup cooked noodles",
            "calories": 221,
            "protein_g": 8.1,
            "carbs_g": 40.3,
            "fat_g": 3.3
        }
    }

    items = []
    unknown_ingredients = []
    totals = {
        "calories": 0,
        "protein_g": 0,
        "carbs_g": 0,
        "fat_g": 0
    }

    for ingredient in normalized_ingredients:
        nutrition = nutrition_data.get(ingredient)

        if nutrition is None:
            unknown_ingredients.append(ingredient)
            continue

        items.append({
            "ingredient": ingredient,
            **nutrition
        })

        for nutrient in totals:
            totals[nutrient] += nutrition[nutrient]

    return {
        "note": "Estimates use one typical serving of each ingredient and may vary by product and preparation method.",
        "items": items,
        "estimated_total": {
            nutrient: round(value, 1)
            for nutrient, value in totals.items()
        },
        "unknown_ingredients": unknown_ingredients
    }

tools = [
    {
        "type": "function",
        "function": {
            "name": "suggest_recipe",
            # The model does not directly run your Python function. It only says, “I want to call suggest_recipe with these arguments.” Your code then runs the actual Python function.
            # This helps the model decide when to use the tool.
            "description": "Suggest recipes based on ingredients the user has.",
            # It tells the model what arguments are valid
            "parameters": {
                "type": "object",
                "properties": {
                    "ingredients": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        # This describes one parameter of the function.
                        # What should go inside the ingredients argument?
                        "description": "Ingredients the user has, such as chicken, rice, egg, tomato."
                    }
                },
                # means the model must include ingredients when calling the function.
                "required": ["ingredients"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_nutrition",
            "description": "Estimate calories and macronutrients for a list of ingredients using typical serving sizes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ingredients": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Ingredients to estimate nutrition for, such as chicken, rice, egg, or tomato."
                    }
                },
                "required": ["ingredients"]
            }
        }
    }
]

MAX_STEPS = 5
TOTAL_TIMEOUT_SECONDS = 30
REQUEST_TIMEOUT_SECONDS = 15

TOOL_FUNCTIONS = {
    "suggest_recipe": suggest_recipe,
    "estimate_nutrition": estimate_nutrition,
}


def run_agent(user_input):
    if not isinstance(user_input, str) or not user_input.strip():
        return {
            "status": "error",
            "error": "User input must be a non-empty string"
        }

    try:
        client = create_client()
    except Exception as error:
        return {
            "status": "error",
            "error": f"Client setup failed: {error}"
        }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful cooking assistant. "
                "Use tools when useful. Your final response must be valid JSON "
                'in this format: {"status": "success", "answer": "...", "data": {}}'
            )
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    start_time = time.monotonic()

    for step in range(MAX_STEPS):
        print(step)
        elapsed = time.monotonic() - start_time
        remaining_time = TOTAL_TIMEOUT_SECONDS - elapsed

        if remaining_time <= 0:
            return {
                "status": "error",
                "error": "Agent timed out"
            }

        try:
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=messages,
                tools=tools,
                response_format={"type": "json_object"},
                max_tokens=1000,
                timeout=min(REQUEST_TIMEOUT_SECONDS, remaining_time)
            )
        except Exception as error:
            return {
                "status": "error",
                "error": f"Model request failed: {error}"
            }

        message = response.choices[0].message
        messages.append(message)
        print(message)

        # No tool call means the model produced its final JSON answer.
        if not message.tool_calls:
            if not message.content:
                return {
                    "status": "error",
                    "error": "Model returned empty content"
                }

            try:
                return json.loads(message.content)
            except json.JSONDecodeError as error:
                return {
                    "status": "error",
                    "error": f"Invalid JSON from model: {error}",
                    "raw_content": message.content
                }

        # Execute every tool call, not only tool_calls[0].
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name

            try:
                arguments = json.loads(tool_call.function.arguments)

                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")

                function = TOOL_FUNCTIONS.get(function_name)

                if function is None:
                    tool_result = {
                        "error": f"Unknown function: {function_name}"
                    }
                else:
                    tool_result = function(**arguments)

            except json.JSONDecodeError as error:
                tool_result = {
                    "error": f"Invalid tool arguments: {error}"
                }
            except Exception as error:
                tool_result = {
                    "error": f"Tool execution failed: {error}"
                }

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result)
            })

    return {
        "status": "error",
        "error": f"Agent exceeded maximum steps ({MAX_STEPS})"
    }


if __name__ == "__main__":
    # Hello, what can you do?
    # I have chicken, rice, egg and onion. What can I cook?
    # Estimate nutrition for chicken and rice.
    # Suggest a recipe and estimate its nutrition using chicken and rice.
    
    user_input = input("You: ")
    result = run_agent(user_input)
    print(json.dumps(result, indent=2, ensure_ascii=False))
