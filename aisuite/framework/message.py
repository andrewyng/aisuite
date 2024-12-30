"""Interface to hold contents of api responses when they do not confirm to the OpenAI style response"""


class Message:
    def __init__(self):
        self.content = None
