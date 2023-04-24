import json
import os
import time

import openai

# Define the timestamp at which we can query Codex (again)
next_complete_time = 0
next_edit_time = 0

# Cooldown time between Codex queries
QUERY_COOLDOWN = 3
N_COOLDOWN = 3


def load_codex_api(pass_file: str = "~/.password.json", key: str = "openai-api-completion") -> str:
    """Load the API from `~/.password.json`"""
    with open(os.path.expanduser(pass_file)) as f:
        content = json.load(f)
        api = content[key]
    return api


def get_codex_edit(
    input_code: str,
    n=1,
    instruction="Fix the errors in this Python code",
    model="code-davinci-edit-001",
    # model="text-davinci-edit-001",
    temperature=0,
    return_codes_only=False,
    max_tolerant_invalid_requests: int = 5,
):
    # Wait for cooldown
    global next_edit_time
    while time.time() < next_edit_time:
        time.sleep(max(0.0, next_edit_time - time.time()))

    # Count this call
    global n_calls_Codex_E, n_outputs_Codex_E
    n_calls_Codex_E += 1
    n_outputs_Codex_E += n

    cnt_invalid_requests = 0

    while True:
        try:
            # Query
            openai.api_key = api_edit
            request_output = openai.Edit.create(
                model=model,
                temperature=temperature,
                input=input_code,
                instruction=instruction,
                n=n,
            )

            # Setup cooldown time
            next_edit_time = time.time() + QUERY_COOLDOWN + n * N_COOLDOWN

            if return_codes_only:
                try:
                    outputed_codes = [choice["text"] for choice in request_output["choices"]]
                    return outputed_codes
                except KeyError as e:
                    raise e
            else:
                return request_output

        except openai.error.RateLimitError:
            print("Rate limited")
            time.sleep(15)
        except openai.error.Timeout:
            print("Timeout")
            time.sleep(10)
        except (openai.error.APIConnectionError, openai.error.APIError, openai.error.ServiceUnavailableError):
            print("API/Service error")
            time.sleep(60)
        except openai.error.InvalidRequestError as e:
            print(f"During calling `generate_fixes`, the following error occurs: {e}")
            cnt_invalid_requests += 1
            if cnt_invalid_requests >= max_tolerant_invalid_requests:
                if return_codes_only:
                    return []
                else:
                    return {"choices": []}
        except KeyError:
            print("KeyError")
            pass


def get_codex_complete(
    prompt: str,
    model="code-davinci-002",
    # model="text-davinci-002",
    n=1,
    temperature=0,
    stop=None,
    presence_penalty=0,
    frequency_penalty=0,
    max_tokens=256,
    logprobs=None,
    max_tolerant_invalid_requests: int = 5,
):
    # Wait for cooldown
    global next_complete_time
    while time.time() < next_complete_time:
        time.sleep(next_complete_time - time.time())

    # Count this call
    global n_calls_Codex_C, n_outputs_Codex_C
    n_calls_Codex_C += 1
    n_outputs_Codex_C += n

    cnt_invalid_requests = 0

    while True:
        try:
            # Query
            openai.api_key = api_completion
            request_output = openai.Completion.create(
                model=model,
                prompt=prompt,
                n=n,
                temperature=temperature,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                max_tokens=max_tokens,
                logprobs=logprobs,
            )

            # Setup cooldown time
            next_complete_time = time.time() + QUERY_COOLDOWN + n * N_COOLDOWN

            if request_output["usage"]["prompt_tokens"] + max_tokens > 7500:
                print(f"Warning, the prompt is too long ({request_output['usage']['prompt_tokens']} tokens).")

            try:
                outputted_completions = [choice["text"] for choice in request_output["choices"]]
                if logprobs is not None:
                    outputted_logprobs = [choice["logprobs"]["token_logprobs"] for choice in request_output["choices"]]
                    return outputted_completions, outputted_logprobs
                else:
                    return outputted_completions
            except KeyError as e:
                raise e

        except openai.error.RateLimitError:
            print("Rate limited")
            time.sleep(15)
        except openai.error.Timeout:
            print("Timeout")
            time.sleep(10)
        except (openai.error.APIConnectionError, openai.error.APIError, openai.error.ServiceUnavailableError):
            time.sleep(60)
        except openai.error.InvalidRequestError as e:
            print(f"During calling `generate_feedback`, the following error occurs: {e}")
            cnt_invalid_requests += 1
            if cnt_invalid_requests > max_tolerant_invalid_requests:
                return []
        except KeyError:
            pass


def get_n_queries_Codex_E():
    return n_calls_Codex_E, n_outputs_Codex_E


def get_n_queries_Codex_C():
    return n_calls_Codex_C, n_outputs_Codex_C


# Load codex api
api_completion = load_codex_api(key="openai-api-completion")
api_edit = load_codex_api(key="openai-api-edit")

# Count the number of calls to Codex
n_calls_Codex_E, n_outputs_Codex_E = 0, 0
n_calls_Codex_C, n_outputs_Codex_C = 0, 0
