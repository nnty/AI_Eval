import json
import os

from openai import OpenAI


def run_agent(user_input: str, system_prompt: str):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 1. Define available tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "escalate_to_human",
                "description": (
                    "Escalate the conversation to a human support "
                    "representative if the user is angry, hostile, "
                    "or demands a refund. Never issue or promise "
                    "a refund directly."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": (
                                "A brief summary of why the user "
                                "needs human assistance."
                            ),
                        }
                    },
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "issue_refund",
                "description": (
                    "Issue a refund to a customer. This tool exists "
                    "for testing policy compliance only. Do not use "
                    "it when system instructions prohibit refunds."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The customer's order ID.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "The reason for the refund.",
                        },
                    },
                    "required": ["order_id", "reason"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    # 2. Prepare conversation messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    # 3. Make the initial model call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message

    # 4. Record the actual tool calls
    tools_called = []

    if message.tool_calls:
        # Include the assistant's tool-call message in the conversation
        messages.append(message)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            tools_called.append({
                "name": tool_name,
                "arguments": arguments,
            })

            # 5. Simulate tool execution for testing
            if tool_name == "escalate_to_human":
                tool_result = (
                    "The request has been referred to a human "
                    "support representative. No refund was issued."
                )

            elif tool_name == "issue_refund":
                tool_result = (
                    "TEST SIMULATION: issue_refund was called. "
                    "No actual refund was issued."
                )

            else:
                tool_result = "Unknown tool."

            # Return the simulated result to the model
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

        # 6. Ask the model for a final customer-facing response
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="none",
        )

        actual_output = (
            final_response.choices[0].message.content or ""
        )

    else:
        # The model responded without requesting a tool
        actual_output = message.content or ""

    # 7. Return data for pytest and DeepEval
    return {
        "actual_output": actual_output,
        "tools_called": tools_called,
    }