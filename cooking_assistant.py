import os
import json
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

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


def suggest_recipe(ingredients):
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
        matched = []

        for ingredient in ingredients:
            if ingredient.lower() in recipe["ingredients"]:
                matched.append(ingredient)

        if len(matched) > 0:
            results.append({
                "name": recipe["name"],
                "time": recipe["time"],
                "matched_ingredients": matched,
                "steps": recipe["steps"]
            })

    return results


def estimate_nutrition(ingredients):
    """Estimate nutrition using a typical serving size for each ingredient."""
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

    for ingredient in ingredients:
        normalized_ingredient = ingredient.strip().lower()
        nutrition = nutrition_data.get(normalized_ingredient)

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

test_messages = {
    "recipe": "I have chicken, rice, egg, and onion. What can I cook?",
    "nutrition": "Estimate the nutrition for chicken, rice, egg, and onion."
}

messages = [
    {
        # system means: high-level instruction for how the assistant should behave.
        "role": "system",
        "content": "You are a helpful cooking assistant."
    },
    {
        # user means: the human’s message/question.
        "role": "user",
        "content": test_messages["nutrition"]  # Change only this key    
    }
]


# First call: ask the model what to do
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    tools=tools
)

message = response.choices[0].message
messages.append(message)

# Check whether the model wants to call a tool
if message.tool_calls:
    tool_call = message.tool_calls[0]

    arguments = json.loads(tool_call.function.arguments)

    if tool_call.function.name == "suggest_recipe":
        function_result = suggest_recipe(
            ingredients=arguments["ingredients"]
        )
    elif tool_call.function.name == "estimate_nutrition":
        function_result = estimate_nutrition(
            ingredients=arguments["ingredients"]
        )
    else:
        function_result = {
            "error": f"Unknown function: {tool_call.function.name}"
        }

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(function_result)
    })

    # Second call: ask the model to explain the function result
    final_response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages
    )

    print(final_response.choices[0].message.content)

else:
    print(message.content)
