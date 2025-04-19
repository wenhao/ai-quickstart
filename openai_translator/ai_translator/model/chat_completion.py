import os

from openai import OpenAI

from openai_translator.ai_translator.book.content import ContentType


class ChatCompletion:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def make_text_prompt(self, text: str, target_language: str) -> str:
        return f"翻译为{target_language}: {text}"

    def make_table_prompt(self, table: str, target_language: str) -> str:
        return f"翻译为{target_language}，以空格和换行符表示表格：\n{table}"

    def translate_prompt(self, content, target_language: str) -> str:
        if content.content_type == ContentType.TEXT:
            return self.make_text_prompt(content.original, target_language)
        elif content.content_type == ContentType.TABLE:
            return self.make_table_prompt(content.get_original_as_str(), target_language)

    def make_request(self, prompt):
        response = self.client.chat.completions.create(
            model='qwen-plus',
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        return response.choices[0].message.content.strip()
