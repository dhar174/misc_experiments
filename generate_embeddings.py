from transformers import AutoTokenizer, AutoModel
import torch
import argparse
import numpy as np
import pandas as pd
import os

import io
from pdfminer.layout import LAParams, LTTextBox, LTChar
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfinterp import PDFPageInterpreter
from latex2mathml import LatexNodes2MathML
import pytesseract
def mean_pooling_original(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
def read_pdf(file_path, text_save_path=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found at: " + file_path)
        return
    if text_save_path and not os.path.exists(os.path.dirname(text_save_path)):
        text_save_path = os.path.dirname(text_save_path)
    text_save_path = os.path.join(os.path.dirname(file_path), os.path.splitext(os.path.basename(file_path))[0] + '.txt')
    resource_manager = PDFResourceManager()
    output_string = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    with open(file_path, 'rb') as f:
        parser = PDFParser(f)
        document = PDFDocument(parser)
        device = PDFPageAggregator(resource_manager, laparams=laparams)
        interpreter = PDFPageInterpreter(resource_manager, device)
        for page in PDFPage.create_pages(document):
            interpreter.process_page(page)
            layout = device.get_result()
            for element in layout:
                if isinstance(element, LTTextBox):
                    text = element.get_text()
                    output_string.write(text)
                elif isinstance(element, LTChar) and element.fontname == 'Math1':
                    mathml = '<math xmlns="http://www.w3.org/1998/Math/MathML">' + element.get_text() + '</math>'
                    output_string.write(mathml)
                
    text = output_string.getvalue()
    output_string.close()


    if text_save_path:
        with open(text_save_path, 'w') as f:
            f.write(text)
        print("Text saved at:", text_save_path)
    return text

import docx2txt

def read_docx(file_path):
    text = docx2txt.process(file_path)
    return text

def read_txt(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return text


from tqdm import tqdm
import re
import latex2mathml
from sympy import preview
from PIL import Image
import io
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output  # Assuming model_output is already just the embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_embeddings(texts, model_name, text_save_path=None, device=None, strategy='mean_pooling'):
    assert texts is not None, "Please provide a text."
    assert model_name is not None, "Please provide a model name."
 
    if device is None:
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
    print("Using device to generate embeddings:", device)
    equation_save_path = None
    if text_save_path and not os.path.exists(os.path.dirname(text_save_path)):
        try:
            #if possible, create the directory up to the filename in the path
            os.makedirs(os.path.dirname(text_save_path[:text_save_path.index('/')]))
            print("Created directory:", os.path.dirname(text_save_path[:text_save_path.index('/')]))

        except:
            text_save_path = text_save_path[:-text_save_path[::-1].index('/')]
            pass
        
        text_save_path = os.path.dirname(text_save_path)
        print("New text save path:", text_save_path)
    if text_save_path:
        equation_save_path = os.path.join(os.path.dirname(text_save_path), os.path.splitext(os.path.basename(text_save_path))[0] + '_equations_only.txt')
        text_save_path = os.path.join(os.path.dirname(text_save_path), os.path.splitext(os.path.basename(text_save_path))[0] + '_final.txt')

    #Check if model_name is a string or a tuple of class objects
    if type(model_name) == str:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    elif type(model_name) == tuple:
        tokenizer = model_name[0]
        model = model_name[1].to(device)



    # Find LaTeX equations in the text and replace them with placeholders
    latex_pattern = re.compile(r'\\begin\{.*?\}(?:.|[\n])*?\\end\{.*?\}|\\\[.*?\\\]|\\\((?:.|[\n])*?\\\)|\$(?:.|[\n])*?\$')
    # Find MathML equations in the text and replace them with placeholders
    mathml_pattern = re.compile(r'<math xmlns="http://www.w3.org/1998/Math/MathML">.*?</math>')
    for text in texts:
        placeholders = {}
        for i, equation in enumerate(re.findall(latex_pattern, text)):
            placeholder = f'[EQUATION_{i}]'
            text = text.replace(equation, placeholder)
            placeholders[placeholder] = equation
        for i, equation in enumerate(re.findall(mathml_pattern, text)):
            if equation in placeholders.values():
                continue
            else:
                placeholder = f'[EQUATION_{i}]'
                text = text.replace(equation, placeholder)
                placeholders[placeholder] = equation

        # Convert LaTeX equations to MathML
        mathml_equations = []
        for equation in placeholders.values():
            mathml = LatexNodes2MathML().latex_to_mathml(equation)
            mathml_equations.append(mathml)

        # Convert MathML to plain text
        plain_text_equations = []
        for mathml in mathml_equations:
            preview(mathml, viewer='file', filename='equation.png', euler=False)
            with Image.open('equation.png') as img:
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                plain_text = pytesseract.image_to_string(buffer, lang='eng', config='--psm 6')
                plain_text_equations.append(plain_text)

        if equation_save_path:
            with open(text_save_path, 'w') as f:
                f.write(text)
            print("Equations saved at:", text_save_path)

        # Replace placeholders with plain text equations
        for i, plain_text in enumerate(plain_text_equations):
            placeholder = f'[EQUATION_{i}]'
            text = text.replace(placeholder, plain_text)
        if text_save_path:
            with open(text_save_path, 'w') as f:
                f.write(text)
            print("Final text saved at:", text_save_path)
    # Tokenize the text
    # inputs = tokenizer(texts, return_tensors="pt", max_length=512, truncation=True).to(device)

    all_sentences = []
    for document in texts:
        document.replace('\n', ' ')

        sentences = document.split('. ')
        
        all_sentences.extend(sentences)
    

    # model_name = 'bert-base-nli-mean-tokens'
    # model = SentenceTransformer(model_name)    


    all_sentences = [s+'. ' for s in all_sentences if len(s) > 5]

    print(f"all_sentences 1 : {all_sentences[1]}")
    print(f"all_sentences 2 : {all_sentences[2]}")
    print(f"all_sentences 3 : {all_sentences[3]}")
    # tokenized_sentences = [tokenizer.encode(s, add_special_tokens=True) for s in all_sentences]
    
    
    embeddings = []
    # tokenized_sentences = []
    model.to(device)
    # model.eval()
    # bot_cortex.search.model.embeddings.word_embeddings.padding_idx = None
    # model.embeddings.word_embeddings.weight.requires_grad = False
    # with torch.no_grad():
    encoded_input = tokenizer(
                all_sentences,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
    for i in tqdm(range(0, len(encoded_input['input_ids']), 1)):
        input_batch = encoded_input['input_ids'][i:i+1]
        attention_mask_batch = encoded_input['attention_mask'][i:i+1]
        outputs = model(input_batch, attention_mask=attention_mask_batch, output_hidden_states=True, return_dict=True)
        embeddings.append(outputs.last_hidden_state)
    #embeddings = torch.cat(embeddings, dim=0)

    embeddings = torch.cat(embeddings, dim=0)
    print(f"embeddings shape : {embeddings.shape}")


    print(f"embeddings shape : {embeddings[0].shape}",
          f"embeddings shape : {embeddings[1].shape}")
    print(f"embeddings type : {type(embeddings)}")
    
    print(f"embeddings type : {type(embeddings)}")
    print(f"embeddings shape : {embeddings.shape}")
    print(encoded_input['attention_mask'].shape)

    attention_mask = encoded_input['attention_mask']

    print(f"attention_mask shape : {attention_mask.shape}")
    if strategy == 'mean':
        embeddings = embeddings.mean(dim=1)
    elif strategy == 'cls':
        embeddings = embeddings[:, 0]
    elif strategy == 'max':
        embeddings = embeddings.max(dim=1)
    elif strategy == 'mean_pooling':
        embeddings = mean_pooling(embeddings, attention_mask.to(device))
    
    
    
    return embeddings, all_sentences



    # embeddings = []
    # for i in tqdm(range(0, len(inputs['input_ids']), batch_size)):
    #     input_batch = inputs['input_ids'][i:i+batch_size]
    #     attention_mask_batch = inputs['attention_mask'][i:i+batch_size]
    #     outputs = model(input_batch, attention_mask=attention_mask_batch)
    #     embeddings.append(outputs.last_hidden_state)
    # embeddings = torch.cat(embeddings, dim=0)
    return embeddings

def main(file_path=None, model_name=None, text_save_path=None, device=None, strategy='mean_pooling'):
    assert file_path is not None, "Please provide a file path."
    if model_name is None:
        model_name = "all-mpnet-base-v2"
    
    
    file_extension = file_path.split('.')[-1]

    if file_extension == 'pdf':
        text = read_pdf(file_path, text_save_path if text_save_path else None)
    elif file_extension == 'docx':
        text = read_docx(file_path)
    elif file_extension == 'txt' or file_extension == 'py':
        text = read_txt(file_path)
    else:
        raise ValueError("File format not supported.")
        return


    embeddings, doc_text = get_embeddings([text], model_name, text_save_path if text_save_path else None, device=device, strategy=strategy)
    print("Embeddings generated with shape:", embeddings.shape)
    #Save the embeddings in several formats
    # torch.save(embeddings, file_path.split('.')[0] + '.pt')
    # torch.save(embeddings, file_path.split('.')[0] + '.pth')
    # torch.save(embeddings, file_path.split('.')[0] + '.bin')
    # np.save(file_path.split('.')[0] + '.npy', embeddings.cpu().detach().numpy())
    # pd.DataFrame(embeddings.cpu().detach().numpy().flatten()).to_csv(file_path.split('.')[0] + '.csv', index=False, header=False)
    # print("Embeddings saved in the following formats:")
    # print("1. PyTorch (.pt)")
    # print("2. PyTorch (.pth)")
    # print("3. PyTorch (.bin)")
    # print("4. Numpy (.npy)")
    # print("5. CSV (.csv)")

    return embeddings, doc_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings for a given text file.')
    parser.add_argument('file_path', type=str, help='Path to the text file')
    parser.add_argument('--embedding_model', type=str, help='Name of the model to use for generating embeddings')
    parser.add_argument('--text_save_path', type=str, help='Path to save the text')
    args = parser.parse_args()

    if args.embedding_model and args.file_path:
        main(args.file_path, args.embedding_model, args.text_save_path if args.text_save_path else None)
    elif args.file_path:
        main(args.file_path,None, args.text_save_path if args.text_save_path else None)
    elif args.embedding_model:
        main(None,args.embedding_model, args.text_save_path if args.text_save_path else None)