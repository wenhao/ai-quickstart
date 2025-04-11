import sys
import os

from openai_translator.ai_translator.model.chat_completion import ChatCompletion
from openai_translator.ai_translator.translator.pdf_translator import PDFTranslator
from openai_translator.ai_translator.utils.argument_parser import ArgumentParser
from openai_translator.ai_translator.utils.config_loader import ConfigLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    argument_parser = ArgumentParser()
    args = argument_parser.parse_arguments()
    config_loader = ConfigLoader(args.config)

    config = config_loader.load_config()

    model_name = args.openai_model if args.openai_model else config['OpenAIModel']['model']
    api_key = args.openai_api_key if args.openai_api_key else config['OpenAIModel']['api_key']
    model = ChatCompletion(model=model_name, api_key=api_key)


    pdf_file_path = args.book if args.book else config['common']['book']
    file_format = args.file_format if args.file_format else config['common']['file_format']

    # 实例化 PDFTranslator 类，并调用 translate_pdf() 方法
    translator = PDFTranslator(model)
    translator.translate_pdf(pdf_file_path, file_format)