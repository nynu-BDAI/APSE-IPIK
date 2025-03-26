
from openai import OpenAI
import base64

class ImageDescriptionGenerator:
    def __init__(self, api_key, base_url):
        # 初始化客户端，指向硅基流动 API
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    def image_to_base64(self, image_path):
        # 将本地图片转换为 Base64
        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_string}"

    def get_image_description(self,base64_image, text_query):

        response = self.client.chat.completions.create(
            model="deepseek-ai/deepseek-vl2",
            #model="doubao-1-5-vision-pro-32k-250115",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_image}
                        },
                        {
                            "type": "text",
                            "text": text_query  # 视觉问答问题
                        }
                    ]
                }
            ],
        )

        # 返回模型生成的文本
        return response.choices[0].message.content




