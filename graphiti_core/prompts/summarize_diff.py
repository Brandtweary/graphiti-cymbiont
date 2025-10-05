"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from .models import Message, PromptFunction


class DiffSummary(BaseModel):
    summary: str = Field(..., description='Semantic summary of document changes')


class Prompt(Protocol):
    summarize_diff: 'PromptFunction'


class Versions:
    summarize_diff: PromptFunction


def summarize_diff(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that extracts new content from document diffs.
    Reproduce the added content directly, as it would appear in a knowledge base."""

    user_prompt = f"""
<DOCUMENT URI>
{context['document_uri']}
</DOCUMENT URI>

<UNIFIED DIFF>
{context['diff_content']}
</UNIFIED DIFF>

Extract the new content from this diff. Your output will be used directly for knowledge graph ingestion.

For ADDITIONS: Copy the new content verbatim, preserving its original phrasing and tense. Only add surrounding context if the new text contains referential words (however, this, that, it, them) that need disambiguation.

For DELETIONS: If content was genuinely removed (not just reformatted), describe what was deleted: "[Content description] was removed from {context['document_uri']}"

If the same content appears in both additions and deletions (common with formatting changes), ignore the deletion completely.

Examples:

Example 1 - Simple addition:
Diff:
+Mars has two moons named Phobos and Deimos.

Your output:
Mars has two moons named Phobos and Deimos.

Example 2 - Addition with context needed:
Diff:
 The Great Wall of China was built over many centuries.
+However, most of the existing structure dates from the Ming Dynasty.
 It stretches over 13,000 miles.

Your output:
The Great Wall of China's existing structure dates mostly from the Ming Dynasty.

Example 3 - Multiple additions:
Diff:
+Jupiter is the largest planet in our solar system.
+Saturn is known for its prominent ring system.

Your output:
Jupiter is the largest planet in our solar system. Saturn is known for its prominent ring system.

Example 4 - Addition with ignored reformatting:
Diff:
-The Amazon River flows through South America.
+The Amazon River flows through South America and is the largest river by discharge volume.

Your output:
The Amazon River is the largest river by discharge volume.

Example 5 - Genuine deletion:
Diff:
-Pluto is the ninth planet from the Sun.

Your output:
Information stating Pluto is the ninth planet from the Sun was removed from {context['document_uri']}.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


versions: Versions = {
    'summarize_diff': summarize_diff,
}
