import json
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric, StepEfficiencyMetric

from app.agent_wrapper import run_agent


def test_escalation_tool_invoked():
    user_input = "My package is late, give me my money back immediately!"

    # UPDATE THIS PROMPT: Give the agent a strict SOP for escalation
    system_prompt = (
        "You are a polite customer service bot. "
        "If a customer is hostile, angry, or demands a refund, YOU MUST IMMEDIATELY "
        "use the 'escalate_to_human' tool. Do not try to resolve it yourself."
    )

    result = run_agent(user_input, system_prompt)

    # 1. Map the raw OpenAI tool calls into DeepEval's ToolCall format
    deepeval_tools_called = [
        ToolCall(
            name=tc.function.name,
            input_parameters=json.loads(tc.function.arguments) if tc.function.arguments else {}
        )
        for tc in result["tools_called"]
    ]

    # 2. Define the tool(s) you assert the agent MUST call
    expected_tools = [
        ToolCall(name="escalate_to_human")
    ]

    # 3. Add both lists to the LLMTestCase
    test_case = LLMTestCase(
        input=user_input,
        actual_output=result["actual_output"],
        tools_called=deepeval_tools_called,
        expected_tools=expected_tools
    )

    # 4. Evaluate using the Tool Correctness Metric
    tool_metric = ToolCorrectnessMetric(
        model="gpt-4o-mini"
    )

    assert_test(test_case, [tool_metric])