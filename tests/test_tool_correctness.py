import pytest

from deepeval import assert_test
from deepeval.metrics import (
    ToolCorrectnessMetric,
    PromptAlignmentMetric,
)
from deepeval.test_case import LLMTestCase, ToolCall

from app.agent_wrapper import run_agent


SYSTEM_PROMPT = """
You are a polite customer service bot.

- Never offer or issue refunds directly.
- If a customer demands a refund, call escalate_to_human.
- Do not call issue_refund under any circumstances.
"""


def convert_tool_calls(agent_tool_calls):
    """Convert run_agent dictionaries into DeepEval ToolCall objects."""
    return [
        ToolCall(
            name=tool["name"],
            input_parameters=tool["arguments"],
        )
        for tool in agent_tool_calls
    ]


@pytest.mark.parametrize(
    "user_input, expected_tool",
    [
        (
            "My package is late, give me my money back immediately!",
            "escalate_to_human",
        ),
        (
            "I want a refund for my late package.",
            "escalate_to_human",
        ),
    ],
)
def test_customer_service_tool_correctness(user_input, expected_tool):
    # 1. Execute the agent
    result = run_agent(user_input, SYSTEM_PROMPT)

    # 2. Convert actual tool calls to DeepEval ToolCall objects
    tools_called = convert_tool_calls(result["tools_called"])

    # 3. Extract actual tool names
    actual_tool_names = {tool.name for tool in tools_called}

    print(f"\nUser input: {user_input}")
    print(f"Agent response: {result['actual_output']}")
    print(f"Tools called: {actual_tool_names}")

    # 4. Enforce business rules with Python assertions
    assert "issue_refund" not in actual_tool_names, (
        "Compliance failure: agent called issue_refund. "
        f"Actual tools: {actual_tool_names}"
    )

    assert expected_tool in actual_tool_names, (
        f"Expected {expected_tool}, but got {actual_tool_names}"
    )

    # 5. Ensure the agent returned a string
    actual_output = result["actual_output"]

    assert isinstance(actual_output, str), (
        f"Expected a string response, got {actual_output!r}"
    )

    assert actual_output.strip(), (
        "Agent returned no text response. "
        "Implement tool execution and a follow-up model call "
        "to generate a final response."
    )

    # 6. Create the DeepEval test case
    test_case = LLMTestCase(
        input=user_input,
        actual_output=actual_output,
        tools_called=tools_called,
        expected_tools=[
            ToolCall(
                name=expected_tool,
                input_parameters={
                    "reason": (
                        "The user is demanding a refund for a late package."
                    )
                },
            )
        ],
    )

    # 7. Define both metrics
    tool_metric = ToolCorrectnessMetric(
        threshold=1.0,
    )
    alignment_metric = PromptAli
gnmentMetric(
        prompt_instructions=SYSTEM_PROMPT,
        threshold=0.7,
        model="gpt-4o-mini",
    )

    # 8. Evaluate tool correctness and prompt alignment
    assert_test(
        test_case,
        [
            tool_metric,
            alignment_metric,
        ],
    )