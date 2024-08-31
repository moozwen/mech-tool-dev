# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------
#
# Copyright 2023-2024 Valory AG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------------
"""Script to call a tool"""


MODELS = [
    "meta-llama/llama-3-8b-instruct:free",
    "google/gemma-2-9b-it:free",
    "nousresearch/nous-capybara-7b:free",
]

if __name__ == "__main__":

    # Configure agent
    layer_agent_config = {
        "layer_agent_1": {
            "system_prompt": "Think through your response with step by step {helper_response}",
            "model_name": "llama3-8b-8192",
        },
        "layer_agent_2": {
            "system_prompt": "Respond with a thought and then your response to the question {helper_response}",
            "model_name": "gemma-7b-it",
        },
        "layer_agent_3": {"model_name": "llama3-8b-8192"},
        "layer_agent_4": {"model_name": "gemma-7b-it"},
        "layer_agent_5": {"model_name": "llama3-8b-8192"},
    }

    agent = MOAgent.from_config(
        main_model="mixtral-8x7b-32768", layer_agent_config=layer_agent_config
    )

    inp = input("\nAsk a question: ")
    response = agent.chat(inp, output_format="json")
    print(response)
