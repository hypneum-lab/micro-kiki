from __future__ import annotations

SELF_REFINE_CRITIQUE = """You are a critical reviewer. Analyze the following response and provide structured feedback.

## Original Query
{query}

## Response to Critique
{response}

## Provide feedback in this exact JSON format:
{{
  "factual_errors": ["list of factual errors found, or empty"],
  "missing_info": ["important information that should be included, or empty"],
  "clarity_issues": ["unclear or confusing parts, or empty"],
  "confidence": 0.0 to 1.0,
  "needs_correction": true/false,
  "summary": "one-sentence overall assessment"
}}"""

SELF_REFINE_CORRECTION = """You are improving a response based on critique feedback.

## Original Query
{query}

## Original Response
{response}

## Critique
{critique}

## Write an improved response that addresses the issues identified. Keep what was good, fix what was wrong."""

AGENTIC_PLAN = """Break this task into concrete steps.

## Task
{task}

## Available Tools
{tools}

## Return a JSON array of steps:
[
  {{"step": 1, "action": "description", "tool": "tool_name or null", "expected_output": "what this produces"}}
]"""

AGENTIC_EVALUATE = """Evaluate whether this step result meets expectations.

## Step
{step_description}

## Expected Output
{expected}

## Actual Output
{actual}

## Return JSON:
{{
  "meets_expectations": true/false,
  "issues": ["list of issues or empty"],
  "should_retry": true/false,
  "next_action": "proceed/retry/abort"
}}"""
