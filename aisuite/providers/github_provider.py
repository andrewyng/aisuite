import os
import time
import json
import requests

from aisuite.provider import Provider, LLMError
from aisuite.framework.chat_completion_response import ChatCompletionResponse
from aisuite.framework.message import Message

class GitHubProvider(Provider):
    """
    GitHubProvider integrates GitHub Copilot's experimental GPT-4-based chat completions.
    It authenticates via GitHub's device code flow and retrieves an internal session token before
    sending chat messages. Access is provided solely through a GitHub Copilot subscription.
    """

    MODEL = 'gpt-4'

    def __init__(self, **config):
        # Optionally override client_id via config.
        self.client_id = config.get('client_id', 'Iv1.b507a08c87ecfe98')
        self.token = None          # Session token for API calls
        self.access_token = None   # GitHub OAuth access token
        self.messages = []         # Conversation history

    def _setup(self):
        # Request device and user codes for authentication.
        resp = requests.post(
            'https://github.com/login/device/code',
            headers={
                'accept': 'application/json',
                'editor-version': 'Neovim/0.6.1',
                'editor-plugin-version': 'copilot.vim/1.16.0',
                'content-type': 'application/json',
                'user-agent': 'GithubCopilot/1.155.0',
                'accept-encoding': 'gzip,deflate,br'
            },
            data=json.dumps({"client_id": self.client_id, "scope": "read:user"})
        )
        resp_json = resp.json()
        device_code = resp_json.get('device_code')
        user_code = resp_json.get('user_code')
        verification_uri = resp_json.get('verification_uri')
        print(f'Please visit {verification_uri} and enter code {user_code} to authenticate.')

        # Poll for OAuth access token.
        while True:
            time.sleep(5)
            resp = requests.post(
                'https://github.com/login/oauth/access_token',
                headers={
                    'accept': 'application/json',
                    'editor-version': 'Neovim/0.6.1',
                    'editor-plugin-version': 'copilot.vim/1.16.0',
                    'content-type': 'application/json',
                    'user-agent': 'GithubCopilot/1.155.0',
                    'accept-encoding': 'gzip,deflate,br'
                },
                data=json.dumps({
                    "client_id": self.client_id,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
                })
            )
            resp_json = resp.json()
            access_token = resp_json.get('access_token')
            if access_token:
                break

        # Save the access token to a file for reuse.
        with open('.copilot_token', 'w') as f:
            f.write(access_token)
        print('Authentication success!')
        self.access_token = access_token

    def _get_token(self):
        # Check if the .copilot_token file exists; if not, run the setup.
        if not os.path.exists('.copilot_token'):
            self._setup()
        else:
            with open('.copilot_token', 'r') as f:
                self.access_token = f.read()
        # Retrieve a session token using the stored access token.
        resp = requests.get(
            'https://api.github.com/copilot_internal/v2/token',
            headers={
                'authorization': f'token {self.access_token}',
                'editor-version': 'Neovim/0.6.1',
                'editor-plugin-version': 'copilot.vim/1.16.0',
                'user-agent': 'GithubCopilot/1.155.0'
            }
        )
        resp_json = resp.json()
        self.token = resp_json.get('token')

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Creates a chat completion using GitHub Copilot's experimental endpoint.
        The provided messages are merged with the conversation history.
        """
        # Ensure a valid session token exists.
        if self.token is None:
            self._get_token()

        # Merge incoming messages into the conversation history.
        if messages:
            # Assuming messages is a list of dicts in the format {"content": ..., "role": ...}
            self.messages.extend(messages)

        try:
            resp = requests.post(
                'https://api.githubcopilot.com/chat/completions',
                headers={
                    'authorization': f'Bearer {self.token}',
                    'Editor-Version': 'vscode/1.80.1',
                },
                json={
                    'intent': False,
                    'model': self.MODEL,
                    'temperature': 0,
                    'top_p': 1,
                    'n': 1,
                    'stream': True,
                    'messages': self.messages
                }
            )
        except requests.exceptions.ConnectionError as e:
            raise LLMError(f"Connection error: {e}")

        result = ''
        # Process the streaming response.
        for line in resp.text.split('\n'):
            if line.startswith('data: {'):
                try:
                    json_completion = json.loads(line[6:])
                    delta = json_completion.get('choices')[0].get('delta', {})
                    completion = delta.get('content')
                    if completion:
                        result += completion
                    else:
                        result += '\n'
                except Exception:
                    continue

        # Append the assistant's response to the conversation history.
        self.messages.append({
            "content": result,
            "role": "assistant"
        })

        if result == '':
            print(resp.status_code)
            print(resp.text)

        # Build and return a ChatCompletionResponse object.
        response_obj = ChatCompletionResponse()
        response_obj.choices[0].message.content = result
        response_obj.choices[0].message.role = "assistant"
        return response_obj
