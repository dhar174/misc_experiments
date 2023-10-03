# Import required libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForQuestionAnswering, AutoConfig
from xformers.components.attention import FavorAttention
import argparse
from generate_embeddings import get_embeddings,  mean_pooling, read_docx, read_txt, read_pdf
from generate_embeddings import main as gen_emb
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.quantization import quantize_dynamic, fuse_modules
import transformers
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, util

from accelerate import infer_auto_device_map, init_empty_weights
from sklearn.decomposition import PCA
import numpy as np
import faiss
from accelerate.utils import get_max_memory, is_bf16_available
import tempfile
from accelerate import cpu_offload
from accelerate import find_executable_batch_size
from deepspeed.ops.transformer.inference import DeepSpeedTransformerInference
from deepspeed.ops.sparse_attention import SparseSelfAttention
from deepspeed import module_inject
import deepspeed.module_inject as module_inject
from transformers import pipeline, CONFIG_MAPPING, MODEL_MAPPING
import os
from transformers.integrations import HfDeepSpeedConfig
from accelerate import Accelerator, DistributedType
import deepspeed
# from transformers.integrations import (
#     is_deepspeed_zero3_enabled,
#     is_accelerate_available,
#     is_deepspeed_available,
# )
from deepspeed import OnDevice


from accelerate import Accelerator, DistributedType
from accelerate import load_checkpoint_and_dispatch, dispatch_model, dispatch_model
def set_deepspeed_activation_checkpointing(deepspeed_config):
    deepspeed.checkpointing.configure(
        None, deepspeed_config=deepspeed_config, partition_activations=True
    )

    deepspeed.checkpointing.partition_activations = True
    deepspeed.checkpointing.cpu_checkpointing = True
    deepspeed.checkpointing.checkpoint_activations = True
    deepspeed.checkpointing.synchronize_checkpoint_boundary = True
    deepspeed.checkpointing.contiguous_memory_optimization = True



def dsconfig(fr):
    ds_config = ""

    if fr == "Large":
        ds_config = {
            "sparse_attention": {
                "mode": "fixed",
                "block": 16,
                "different_layout_per_head": True,
                "num_local_blocks": 4,
                "num_global_blocks": 1,
                "attention": "bidirectional",
                "horizontal_global_attention": False,
                "num_different_global_patterns": 4,
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "amp": {"enabled": True, "opt_level": "auto"},
            "bf16": {"enabled": False},
            "zero_optimization": {
                "stage": 3,
                "offload_param": {"device": "cpu", "pin_memory": True},
                "overlap_comm": True,
                "contiguous_gradients": True,
                "allgather_bucket_size": 1e7,
                "reduce_bucket_size": 1e7,
                "stage3_prefetch_bucket_size": 1e7,
                "stage3_max_live_parameters": 5e8,
                "stage3_max_reuse_distance": 5e8,
                "stage3_param_persistence_threshold": 1e5,
                "stage3_gather_16bit_weights_on_model_save": True,
            },
            "steps_per_print": 2000,
            "sub_group_size": 5e8,
            "train_batch_size": 1,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False,
        }
    elif fr == "Medium":
        ds_config = {
            "sparse_attention": {
                "mode": "fixed",
                "block": 16,
                "different_layout_per_head": True,
                "num_local_blocks": 4,
                "num_global_blocks": 1,
                "attention": "bidirectional",
                "horizontal_global_attention": False,
                "num_different_global_patterns": 4,
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "amp": {"enabled": True, "opt_level": "auto"},
            "bf16": {"enabled": False},
            "zero_optimization": {
                "stage": 3,
                "offload_param": {"device": "cpu", "pin_memory": True},
                "overlap_comm": True,
                "contiguous_gradients": True,
                "allgather_bucket_size": 1e9,
                "reduce_bucket_size": 1e9,
                "stage3_prefetch_bucket_size": 1e9,
                "stage3_max_live_parameters": 5e8,
                "stage3_max_reuse_distance": 5e8,
                "stage3_param_persistence_threshold": 1e6,
                "stage3_gather_16bit_weights_on_model_save": True,
            },
            "steps_per_print": 2000,
            "sub_group_size": 5e8,
            "train_batch_size": 1,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False,
            }
    
    return ds_config
# Function for Linear Projection
def linear_projection(tensor, output_shape):
    input_shape = tensor.numel()
    linear_layer = nn.Linear(input_shape, output_shape)
    flattened_tensor = tensor.view(1, -1)
    return linear_layer(flattened_tensor)

# Function for Average Pooling
def average_pooling(tensor, output_shape):
    factor = tensor.numel() // output_shape
    avg_pool = nn.AvgPool1d(factor, stride=factor)
    pooled_tensor = avg_pool(tensor)
    return pooled_tensor.view(1, -1)[:,:output_shape]

# Function for Custom Projection (Example: sum along dimensions)
def custom_projection(tensor, output_shape):
    sum_tensor = torch.sum(tensor, dim=[1, 2])
    return sum_tensor.view(1, -1)[:,:output_shape]

# Function for PCA Projection
def pca_projection(tensor, output_shape):
    tensor_np = tensor.cpu().detach().numpy().reshape(-1)
    pca = PCA(n_components=output_shape)
    projected_tensor = pca.fit_transform(tensor_np.reshape(1, -1))
    return torch.tensor(projected_tensor, dtype=torch.float32)

def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
# Function to download and load the Llama2 model and tokenizer
def load_llama2_model(model_name="meta-llama/Llama-2-7b-chat-hf", quantize=False, device='cpu'):
    """
    Downloads and loads the Llama2 model and tokenizer.
    :param model_name: The name of the model to download. Default is "meta-llama/Llama-2-7b".
    :return: tokenizer, model 
    """


    if quantize:
        print("Quantizing model")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            # Download and load the tokenizer
           

            
            print("Quantized model loaded")
            return tokenizer, model
        except Exception as e:
            print("Error loading quantized model, method 1: ", e)
            try:
                # Download and load the model
                model = AutoModelForCausalLM.from_pretrained(model_name)
                model.eval()
                model.fuse_model()
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model, inplace=True)
                model = torch.quantization.convert(model, inplace=True)
                print("Quantized model 2 loaded")
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                return tokenizer, model
            except Exception as e:
                print("Error loading quantized model, method 2: ", e)

    # try:
    #     # Download and load the tokenizer
    #     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
    #     # Download and load the model
    #     model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
    #     print("Model loaded on GPU")
    #     return tokenizer, model
    # except Exception as e:
    #     print("Error loading model on GPU: ", e)
    model = None
    tokenizer = None
    if device == 'cuda':
        try:
            with torch.no_grad():
                with OnDevice(dtype="auto", device="meta"):
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        offload_state_dict=True,
                        output_attentions=True,
                        output_hidden_states=True,
                        low_cpu_mem_usage=False,
                        torch_dtype="auto",
                        offload_folder="/home/darf3/buddy/experiments/offload",
                    ).to(device)
                print("Model loaded on GPU")
        except Exception as e:
            print("Error loading model on GPU: ", e)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding=True, truncation=True)

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding=True, truncation=True)
        # Download and load the model
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        print("Model loaded on CPU")

    print(f"Model loaded on {device}")
    return tokenizer, model

# def load_sentence_transformer_model(model_name="paraphrase-distilroberta-base-v1"):
#     """
#     Downloads and loads the Sentence Transformer model.
#     :param model_name: The name of the model to download. Default is "paraphrase-distilroberta-base-v1".
#     :return: model 
#     """
#     try:
#         # Download and load the model
#         model = SentenceTransformer(model_name)
#         return model
#     except Exception as e:
#         print("Error loading model: ", e)
#         return None


def FAISS_embeddings(embeddings, query, model, tokenizer):
    """
    Uses FAISS to compute the similarity between the query and each sentence in the document.
    The sentence with the highest similarity score is used as the context for the QA model to generate an answer.
    :param embeddings: The embeddings.
    :param query: The query.
    :return: answer
    """
    try:
        # Create an index for the embeddings
        index = faiss.IndexFlatIP(embeddings.shape[-1])
        index.add(embeddings)

        # Compute the similarity between the query and each sentence in the document
        similarities, indices = index.search(query, 1)

        # Get the index of the sentence with the highest similarity score
        max_index = indices[0][0]

        # Use the sentence with the highest similarity score as the context for the QA model
        context = embeddings[max_index]

        # Generate an answer using the autoregressive decoding method
        with torch.no_grad():
            output = model.generate(input_ids=context.unsqueeze(0))

        # Decode the output IDs to get the answer text
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        print("Error generating answer: ", e)
        return e

def xformers_attention_layer(query, embeddings):
    try:
        attention_layer = FavorAttention(
            dim_features= 1024,
            heads=4,
            causal=True,
            generalized_attention=True,
            kernel_fn="relu",
            qr_uniform_q=True,
            qr_uniform_d=True,
            dropout=0.1,
            no_projection=True,

        )
        # query = query.mean(dim=1)  # Using mean as a representation
        # use the same query for all embeddings
        
        print(query.shape)
        print(embeddings.shape)
        embeddings = embeddings.clone().detach()
        scores = []
        top_k = 3
        projected_query = linear_projection(query, 1024)
        for i in range(len(embeddings[0])):
            print(embeddings[0][i].shape)
            scores[i] = attention_layer(projected_query, embeddings[0][i], embeddings[0][i])
            
        top_results = torch.topk(scores, k=top_k)
        print(top_results)
        return scores
    except Exception as e:
        print("Attention layer failed with error: ", e)
        print(f"Offending line: {e.__traceback__.tb_lineno}")
        return e

def attention_mechanism(embeddings, query):
    try:
        
        
        weighted_embeddings = xformers_attention_layer(query, embeddings)
        return weighted_embeddings
    except Exception as e:
        print("Attention mechanism failed with error: ", e)
        print(f"Offending line: {e.__traceback__.tb_lineno}")
        return e
    weighted_embeddings = xformers_attention_layer(query, embeddings)
    return weighted_embeddings

# Function to run the model and get predictions
def run_model(tokenizer, model, text, device='cpu'):
    """
    Runs the Llama2 model and returns the predictions.
    :param tokenizer: The tokenizer object.
    :param model: The model object.
    :param text: The input text string.
    :return: predictions
    """
    # Tokenize the input text
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    
    # Run the model
    outputs = model.generate(inputs, max_length=136, do_sample=True)
    
    # Get the predictions
    predictions = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return predictions





# Function to load the PyTorch embeddings from a .pt file
def load_embeddings(embeddings_path):
    """
    Load embeddings from a PyTorch .pt file.
    
    Parameters:
        embeddings_path (str): The path to the .pt file containing the embeddings.
    
    Returns:
        torch.Tensor: The loaded embeddings.
    """
    if embeddings_path and not os.path.exists(os.path.dirname(embeddings_path)):
        try:
            #if possible, create the directory up to the filename in the path
            os.makedirs(os.path.dirname(embeddings_path[:embeddings_path.index('/')]))
            print("Created directory:", os.path.dirname(embeddings_path[:embeddings_path.index('/')]))

        except:
            try:
                embeddings_path = embeddings_path[:-embeddings_path[::-1].index('/')]
            except:
                print("Trying again to find embeddings file")
                #grab filename if no directory was found in embeddings_path
                embeddings_path = embeddings_path[:-embeddings_path[::-1].index('\\')] if '\\' in embeddings_path else embeddings_path
                embeddings_path = os.path.join(os.getcwd(), embeddings_path)
            pass
        print("New text save path:", embeddings_path)

    
    try:
        embeddings = torch.load(embeddings_path)
    except Exception as e:
        print(f"Embeddings file {embeddings_path} not found or corrupted, error {e}")
        return None
    return embeddings


# Function to preprocess and tokenize the query text
def preprocess_and_tokenize_query(query, tokenizer):
    """
    Preprocess and tokenize a query for the QA model.
    
    Parameters:
        query (str): The query text.
        tokenizer: The tokenizer compatible with the QA model.
        
    Returns:
        torch.Tensor: The tokenized query.
    """
    inputs = tokenizer(query, return_tensors='pt', max_length=512, truncation=True)
    return inputs


# Function to run QA inference using an autoregressive model like Llama-2-7b-chat-hf
def run_qa_inference(model, tokenizer, document_text,embeddings, tokenized_query, embeddings_model=None, embeddings_tokenizer=None, device='cpu'):
    """
    Run QA inference using the loaded embeddings and tokenized query.
    
    Parameters:
        model: The pre-loaded autoregressive model.
        tokenizer: The tokenizer compatible with the model.
        embeddings (torch.Tensor): The loaded embeddings.
        tokenized_query (torch.Tensor): The tokenized query.
        
    Returns:
        str: The model's answer to the query.
    """
    # Combine the loaded embeddings and tokenized query
    # In a real-world scenario, the embeddings could be used in a more complex manner
    query = tokenized_query
    print("Method 1")
    answer = None
    answers = []
    # try:
    #     print(embeddings.shape)
    #     print(type(tokenized_query))
    #     print(tokenized_query['input_ids'].shape)
    #     for i in range(len(embeddings)):
            
    #         # tokenized_query['input_ids'] is shape torch.Size([1, 9])
    #         # embeddings[i] is shape torch.Size([1, 512])
           
    #         combined_input_ids = torch.stack((embeddings[i], tokenized_query['input_ids']), dim=-1)

    #         combined_input_ids = tokenizer(combined_input_ids, return_tensors='pt')

    #         # Generate an answer using the autoregressive decoding method
    #         with torch.no_grad():
    #             output = model.generate(input_ids=combined_input_ids)
            
    #         # Decode the output IDs to get the answer text
    #         answer = tokenizer.decode(output[0], skip_special_tokens=True)
    #         answers[i] = answer
    #     print(f"Answers to the question {tokenizer.decode(tokenized_query['input_ids'], skip_special_tokens=True)} based on the document {embeddings}: {answers} using method 1 \n\n")

    # except Exception as e:
    #     print("Method 1 failed with error: ", e)
    #     print(f"Offending line: {e.__traceback__.tb_lineno}")

    # print("Method 2")
    # try:
    #     input_ids = embeddings.clone().detach()
    #     query_input_ids = tokenized_query['input_ids']
    #     print(input_ids.shape)
    #     print(query_input_ids.shape)
    #     # if len(input_ids.shape) != len(query_input_ids.shape):
    #     #     # Reshape the tensors to have the same number of dimensions
    #     #     if len(input_ids.shape) > len(query_input_ids.shape):
    #     #         query_input_ids = query_input_ids.unsqueeze(0)
    #     #     else:
    #     #         input_ids = input_ids.unsqueeze(0)


    #     # Initialize your tensor

    #     # Flatten the tensor
    #     flattened_tensor = input_ids.flatten()

    #     # Create a linear layer for projection
    #     linear_layer = nn.Linear(512 * 4096, 1024)

    #     # Project the tensor
    #     projected_tensor = linear_layer(flattened_tensor)

    #     print(projected_tensor.shape)
    #     print(input_ids.shape)
    #     print(flattened_tensor.shape)
    #     print(query_input_ids.shape)

    #     combined_input_ids = torch.cat((input_ids, query_input_ids), dim=0)

    #     # Generate an answer using the autoregressive decoding method
    #     with torch.no_grad():
    #         output = model.generate(input_ids=combined_input_ids)
        
    #     # Decode the output IDs to get the answer text
    #     answer = tokenizer.decode(output[0], skip_special_tokens=True)
    #     print(f"Answer to the question {tokenizer.decode(tokenized_query['input_ids'], skip_special_tokens=True)} based on the document {embeddings}: {answer} using method 2 \n\n")
    # except Exception as e:
    #     print("Method 2 failed with error: ", e)
    #     print(f"Offending line: {e.__traceback__.tb_lineno}")


    # print("Method 3")
    # try:
    #     ids_with_attention = attention_mechanism(embeddings, tokenized_query['input_ids'])
    #     answer = tokenizer.decode(ids_with_attention, skip_special_tokens=True)
    #     print(f"Answer to the question {tokenizer.decode(tokenized_query['input_ids'], skip_special_tokens=True)} based on the document {embeddings}: {answer} using method 3 \n\n")
    # except Exception as e:
    #     print("Attention mechanism, Method 3, failed with error: ", e)
    #     print(f"Offending line: {e.__traceback__.tb_lineno}")
        
    # print("Method 4")
    # try:
    #     inputs = tokenized_query['input_ids']

    #     # Concatenate the embeddings with the encoded input
    #     if type(embeddings) == torch.Tensor:
    #         embeddings_tensor = embeddings.clone().detach()  # If the embeddings are a tensor, use them directly
    #     else:
    #         embeddings_tensor = torch.Tensor(embeddings).clone().detach()  # If the embeddings are a list, convert them to a tensor
    #     inputs = torch.tensor(inputs, dtype=torch.long)
        
    #     inputs = torch.cat((inputs, embeddings_tensor.flatten().unsqueeze(0)), dim=-1)
    #     print(inputs.shape)
    #     print(embeddings_tensor.flatten().unsqueeze(0).shape)
    #     # Use a linear layer to project the embeddings to the same dimension as the model's embeddings
    #     projection_layer = nn.Linear(embeddings_tensor.shape[1], 1024)  

    #     # Get the projection layer's weight tensor
    #     projection_weight = projection_layer.weight.t()  # Transpose the weight tensor
    #     # Reshape the input embeddings to have the same number of dimensions as the projection layer's weight tensor
    #     inputs = inputs.unsqueeze(0)
    #     print(inputs.shape)


    #     print(projection_weight.shape)
    #     # Project the embeddings to the same dimension as the model's embeddings by multiplying them by the projection layer's weight tensor
    #     # inputs = torch.matmul(inputs, projection_weight)  # Perform the matrix multiplication
    #     inputs = projection_layer(inputs)
    #     # Now the embeddings have the same dimension as the model's embeddings, which is 1024, so we can use them directly as input to the model


    #     # Generate the output
    #     output = model.generate(inputs, max_length=1024, pad_token_id=tokenizer.eos_token_id)

    #     # Decode the output
    #     answer = tokenizer.decode(output[0], skip_special_tokens=True)
    #     print(f"Answer to the question {tokenizer.decode(tokenized_query['input_ids'], skip_special_tokens=True)} based on the document {embeddings}: {answer} using method 4 \n\n")
        
    # except Exception as e:
    #     print("Method 4 failed with error: ", e)
    #     print(f"Offending line: {e.__traceback__.tb_lineno}")
    # print("Method 5")
    # #AKA the simple and easy method
    # try:
    #     #query = query + " " + embeddings
    #     query = torch.tensor(tokenized_query['input_ids']).clone().detach()
    #     query = torch.stack((query, embeddings.clone().detach()), dim=-1)
    #     query = torch.tensor(query).unsqueeze(0).to_string()
    #     answer = run_model(tokenizer, model, query)
    #     print(f"Answer to the question {tokenizer.decode(tokenized_query['input_ids'], skip_special_tokens=True)} based on the document {embeddings}: {answer} using method 5 \n\n")
    # except Exception as e:
    #     print("Method 5 failed with error: ", e)
    #     print(f"Offending line: {e.__traceback__.tb_lineno}")
    
    # print("Method 6")
    # # Method 6
    # # This method is the only other way I can think of to use the embeddings
    # # The embeddings are used to generate a response, which is then used as the input to the QA model to generate an answer
    # try:
    #     # Generate a response using the embeddings
    #     embeddings_input = embeddings.clone().detach()
    #     embeddings_input = embeddings_input[0].tolist()
    #     for i in range(len(embeddings_input[0])):
    #         embeddings_input[0][i] = embeddings_input[0][i]
    #     # embeddings_input = torch.tensor(embeddings_input)
    #     print(type(embeddings_input))
    #     response = run_model(tokenizer, model, embeddings_input)
    #     print(f"Response: {response}")
        
    #     # Combine the response with the query
    #     query = tokenized_query
    #     query = query.join(response)
    #     query = torch.tensor(tokenizer.encode(query, return_tensors="pt"))
        
    #     # Generate an answer using the autoregressive decoding method
    #     with torch.no_grad():
    #         output = model.generate(input_ids=query)
        
    #     # Decode the output IDs to get the answer text
    #     answer = tokenizer.decode(output[0], skip_special_tokens=True)
    #     print(f"Answer to the question {tokenizer.decode(tokenized_query['input_ids'], skip_special_tokens=True)} based on the document {embeddings}: {answer} using method 6 \n\n")
    # except Exception as e:
    #     print("Method 6 failed with error: ", e)
    #     print(f"Offending line: {e.__traceback__.tb_lineno}")

    # print("Method 7")
    # # Method 7
    # # This method is the final foolproof method. Pretty genius if I do say so myself... It's not based on any of the previous methods, I really had to think outside the box for this one.
    # # Think about it some more, and you'll see why it works.
    # try:
    #     # The answer to any question is 42
    #     pass
    #     # query = "What is the answer to life, the universe, and everything?"
    #     # answer = "42"
    #     # print(f"Answer to the question {tokenizer.decode(tokenized_query['input_ids'], skip_special_tokens=True)} based on the document {embeddings}: {answer} using method 7 \n\n")
    # except Exception as e:
    #     print("Method 7 failed with error: ", e)
    #     print(f"Offending line: {e.__traceback__.tb_lineno}")
        
    # print("Method 8")
    # # Method 8
    # # This method uses the embeddings to weight the importance of each token in the tokenized query
    # # The weighted query is then used as input to the QA model to generate an answer
    # try:
    #     # Get the embeddings for each token in the tokenized query
    #     query_embeddings = []
    #     q = tokenized_query['input_ids'].tolist()
    #     for token in q:
    #         token_index = q.index(token)
    #         print(token_index)
    #         print(q[token_index])
    #         print(torch.tensor(q[token_index], dtype=torch.long).shape)
    #         print(tokenizer.decode(torch.tensor(q[token_index], dtype=torch.long), skip_special_tokens=True))
    #         print(embeddings.where(torch.tensor(q[token_index] == token), torch.tensor(q[token_index], dtype=torch.long)).shape)
    #         print(embeddings.where(torch.tensor(q[token_index] == token), torch.tensor(q[token_index], dtype=torch.long)))
    #         print(token.shape)
    #         token_embedding = embeddings.where(torch.tensor(q[token_index] == token), torch.tensor(q[token_index], dtype=torch.long))
    #         query_embeddings.append(token_embedding)
    #     query_embeddings = torch.stack(query_embeddings)
    #     #Make sure the embeddings are a tensor and not strings
    #     query_embeddings = torch.tensor(query_embeddings)

    #     # Compute the attention weights for each token in the query
    #     attention_weights = torch.matmul(query_embeddings, embeddings.permute(0, 2, 1))
    #     attention_weights = torch.softmax(attention_weights, dim=-1)


    #     # Weight the tokenized query using the attention weights
    #     weighted_query = torch.matmul(attention_weights, query_embeddings)

    #     # Generate an answer using the autoregressive decoding method
    #     with torch.no_grad():
    #         output = model.generate(input_ids=weighted_query.unsqueeze(0))

    #     # Decode the output IDs to get the answer text
    #     answer = tokenizer.decode(output[0], skip_special_tokens=True)
    #     print(f"Answer to the question {tokenizer.decode(tokenized_query['input_ids'], skip_special_tokens=True)} based on the document {embeddings}: {answer} using method 8 \n\n")
    # except Exception as e:
    #     print("Method 8 failed with error: ", e)
    #     print(f"Offending line: {e.__traceback__.tb_lineno}")

    # print("Method 9")
    # # Method 9
    # # This method uses the embeddings to compute the similarity between the query and each sentence in the document
    # # The sentence with the highest similarity score is used as the context for the QA model to generate an answer
    # try:
    #     # Compute the similarity between the query and each sentence in the document
    #     similarities = []
    #     for sentence in embeddings:
    #         print(sentence)
    #         print(sentence.type())
    #         sentence = torch.tensor(sentence).clone().detach()
            
    #         print(sentence.shape)
    #         print(tokenized_query['input_ids'].shape)
            
    #         print(tokenized_query['input_ids'].type())
    #         print(sentence.size())
    #         similarity = util.cos_sim(sentence, tokenized_query['input_ids'])[0]
    #         similarities.append(similarity)
    #     similarities = torch.cat(similarities)
    #     top_results = torch.topk(cos_scores, k=top_k)
    #     print(top_results)

    #     # Get the index of the sentence with the highest similarity score
    #     max_index = torch.argmax(similarities)
    #     print(max_index)
    #     # Use the sentence with the highest similarity score as the context for the QA model
    #     context = embeddings[max_index]

    #     # Generate an answer using the autoregressive decoding method
    #     with torch.no_grad():
    #         output = model.generate(input_ids=context.unsqueeze(0))

    #     # Decode the output IDs to get the answer text
    #     answer = tokenizer.decode(output[0], skip_special_tokens=True)
    #     print(f"Answer to the question {tokenizer.decode(tokenized_query['input_ids'], skip_special_tokens=True)} based on the document {embeddings}: {answer} using method 9 \n\n")
    # except Exception as e:
    #     print("Method 9 failed with error: ", e)
    #     print(f"Offending line: {e.__traceback__.tb_lineno}")
    
    # print("Method 10")
    # # Method 10
    # # This method uses the embeddings to compute the similarity between the query and each sentence in the document
    # # The top-k sentences with the highest similarity scores are concatenated and used as the context for the QA model to generate an answer
    # try:
    #     # Compute the similarity between the query and each sentence in the document
    #     similarities = []
    #     for sentence in embeddings:
    #         print(sentence.type())
    #         sentence = torch.tensor(sentence, dtype=torch.long)
    #         similarity = cosine_similarity(embeddings[sentence], tokenized_query['input_ids'])
    #         similarities.append(similarity)
    #     similarities = torch.cat(similarities)

    #     # Get the indices of the top-k sentences with the highest similarity scores
    #     k = 3
    #     top_k_indices = torch.argsort(similarities, descending=True)[:k]

    #     # Concatenate the top-k sentences to form the context for the QA model
    #     context = ""
    #     for i in top_k_indices:
    #         context += embeddings[i] + " "

    #     # Generate an answer using the autoregressive decoding method
    #     with torch.no_grad():
    #         output = model.generate(input_ids=context.unsqueeze(0))

    #     # Decode the output IDs to get the answer text
    #     answer = tokenizer.decode(output[0], skip_special_tokens=True)
    #     print(f"Answer to the question {tokenizer.decode(tokenized_query['input_ids'], skip_special_tokens=True)} based on the document {embeddings}: {answer} using method 10 \n\n")
    # except Exception as e:
    #     print("Method 10 failed with error: ", e)
    #     print(f"Offending line: {e.__traceback__.tb_lineno}")
    
    # print("Method 11")
    # # Method 11
    # # This method uses a transformer language model to generate a response to the query, which is then used as the input to the QA model to generate an answer
    # try:
    #     query = tokenized_query['input_ids']
    #     print(query)
    #     print(query.shape)
    #     print(query.type())
    #     # Generate a response to the query using the transformer language model
    #     response = model.generate(input_ids=query, max_length=136, pad_token_id=tokenizer.eos_token_id)
    #     print(response)
    #     print(response.shape)
    #     print(response.type())
    #     if response.shape[0] > 1:
    #         response = torch.tensor(tokenizer.encode(response, return_tensors="pt"))
    #     # Combine the response with the query
    #     query = torch.tensor(query)
    #     query = torch.cat((query, response), dim=-1)

    #     # Generate an answer using the autoregressive decoding method
    #     with torch.no_grad():
    #         output = model.generate(input_ids=query, max_length=136, pad_token_id=tokenizer.eos_token_id)

    #     # Decode the output IDs to get the answer text
    #     answer = tokenizer.decode(output[0], skip_special_tokens=True)
    #     print(f"Answer to the question {query} based on the document {embeddings}: {answer} using method 11 \n\n")
    # except Exception as e:
    #     print("Method 11 failed with error: ", e)
    #     print(f"Offending line: {e.__traceback__.tb_lineno}")

    # print("Method 12")
    # # Method 12
    # # This method uses the embeddings to compute the similarity between the query and each sentence in the document
    # # If the highest similarity score is below a certain threshold, the method returns a default answer
    # # Otherwise, the method uses the sentence with the highest similarity score as the context for the QA model to generate an answer
    # try:
    #     # Compute the similarity between the query and each sentence in the document
    #     similarities = []
    #     for sentence in embeddings:
    #         sentence = torch.tensor(sentence, dtype=torch.long)
    #         similarity = cosine_similarity(embeddings.tolist().index(embeddings[sentence]), tokenized_query['input_ids'])
    #         similarities.append(similarity)
    #     similarities = torch.cat(similarities)

    #     # Get the index of the sentence with the highest similarity score
    #     max_index = torch.argmax(similarities)

    #     # Check if the highest similarity score is below a certain threshold
    #     threshold = 0.5
    #     if similarities[max_index] < threshold:
    #         answer = "I'm sorry, I don't know the answer to that question."
    #     else:
    #         # Use the sentence with the highest similarity score as the context for the QA model
    #         context = embeddings[max_index]

    #         # Generate an answer using the autoregressive decoding method
    #         with torch.no_grad():
    #             output = model.generate(input_ids=context.unsqueeze(0))

    #         # Decode the output IDs to get the answer text
    #         answer = tokenizer.decode(output[0], skip_special_tokens=True)
    #     print(f"Answer to the question {query} based on the document {embeddings}: {answer} using method 12 \n\n")
    # except Exception as e:
    #     print("Method 12 failed with error: ", e)
    #     print(f"Offending line: {e.__traceback__.tb_lineno}")

    # print("Method 13")
    # # Method 13
    # try:
    #     query   = tokenized_query['input_ids']
      

    #     # Create an embedding layer that maps tokens to vectors
    #     embedding_layer = nn.Embedding(len(tokenizer), embeddings.shape[-1])

    #     # Create a tensor of token indices for the query
    #     # Assume that the query is already tokenized and converted to indices
    #     query = torch.tensor(query, dtype=torch.long)

    #     # Get the query embeddings by passing the query tensor to the embedding layer
    #     query_embeds = embedding_layer(query)

    #     # Create a tensor of embeddings of shape (512, 4096)
    #     # Assume that the embeddings are already computed
    #     embeds = embeddings

    #     # Concatenate the query embeddings with the embeddings along the first dimension
    #     # The result will have a shape of (513, 4096)
    #     combined = torch.cat((query_embeds.unsqueeze(0), embeds.unsqueeze(0)), dim=0)
    #     combined = combined.unsqueeze(0)
        
    #     answer = run_model(tokenizer, model, combined)
    #     print(f"Answer to the question {query} based on the document {embeddings}: {answer} using method 13 \n\n")  
    # except Exception as e:
    #     print("Method 13 failed with error: ", e)
    #     print(f"Offending line: {e.__traceback__.tb_lineno}")
    

    print("Method 14")
    # Method 14
    # FAISS is a library for efficient similarity search and clustering of dense vectors
    # This method uses FAISS to compute the similarity between the query and each sentence in the document
    # The sentence with the highest similarity score is used as the context for the QA model to generate an answer
    try:
        # Create an index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        print(embeddings.shape)

        # Add the embeddings to the index
        index.add(embeddings.detach().numpy())

        # Compute the similarity between the query and each sentence in the document
        _, indices = index.search(query.numpy(), k=5)
        relevant_sentence = document_text[int(indices[0][0])]
        
        first_5_sentences = document_text[:5]
        print(f"Query: {query}")
        print(f"Document first sentence: {document_text[0]}")
        print(f"Document second sentence: {document_text[1]}")
        print(f"Document third sentence: {document_text[2]}")
        print(f"Document fourth sentence: {document_text[3]}")
        print(f"Document fifth sentence: {document_text[4]}")

        print(f"Relevant sentence: {relevant_sentence}")
        print(f"Relevant sentence two: {document_text[int(indices[0][1])]}")
        print(f"Relevant sentence three: {document_text[int(indices[0][2])]}")
        print(f"Relevant sentence four: {document_text[int(indices[0][3])]}")
        print(f"Relevant sentence five: {document_text[int(indices[0][4])]}")
        
        # attention_weights = np.softmax(query_embedding)
        # weighted_query = np.sum(attention_weights * query_embedding)

        # # Get the index of the sentence with the highest similarity score
        # max_index = indices[0]

        # # Use the sentence with the highest similarity score as the context for the QA model
        # context = embeddings[max_index]

        # # Generate an answer using the autoregressive decoding method
        # with torch.no_grad():
        #     output = model.generate(input_ids=context.unsqueeze(0))
       
        first_5_sentences = " ".join(first_5_sentences)
        relevant_sentence = tokenizer.encode(first_5_sentences, return_tensors="pt").to(device)

        print(relevant_sentence)
        with torch.no_grad():
            output = model.generate(input_ids=relevant_sentence, max_length=256, pad_token_id=tokenizer.eos_token_id)
            




        # Decode the output IDs to get the answer text
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Answer to the question {query} based on the document: {answer} using method 14 \n\n")
        return answer
    except Exception as e:
        print("Method 14 failed with error: ", e)
        print(f"Offending line: {e.__traceback__.tb_lineno}")

        return e







if __name__ == "__main__":
    transformers.utils.logging.set_verbosity_info()
    

    accelerator = Accelerator()
    ds_config = dsconfig("Large")
    # set_deepspeed_activation_checkpointing(ds_config)

    dschf = HfDeepSpeedConfig(ds_config)


    answered = False
    prev_query = None
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Llama2 text prediction script")
    parser.add_argument("--model", type=str, help="Model name to use for prediction")
    
    parser.add_argument("--input", type=str, help="Input text for prediction")
    parser.add_argument("--file_path", type=str, help="Path to the file containing the text to be embedded")
    parser.add_argument("--embedding_model", type=str, help="Name of the model to use for generating embeddings")
    parser.add_argument("--text_save_path", type=str, help="Path to save the text")
    parser.add_argument("--embeddings_path", type=str, help="Path to the embeddings file")
    parser.add_argument("--query", type=str, help="Query for the QA model")
    parser.add_argument("--quantize", type=bool, help="Quantize the model")
    parser.add_argument("--device", type=str, help="Device to use for prediction")
    model_name = "llama2-7b-chat-hf"
    args = parser.parse_args()
    if parser.parse_args().device:
        device = parser.parse_args().device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if args.model:
        # Load the model and tokenizer
        tokenizer, model = load_llama2_model(args.model, args.quantize, device=device)
        model_name = args.model
    else:
        # Load the default model and tokenizer
        tokenizer, model = load_llama2_model(quantize=args.quantize, device=device)
    embeddings = None
    embedding_name = "none"
    embeddings_model = None
    embeddings_tokenizer = None
    query = None
    if args.query:
        query = args.query
    if args.embedding_model and args.file_path:
        embeddings, doc_text =gen_emb(args.file_path, args.embedding_model, device=device)
        embeddings_model = AutoModel.from_pretrained(args.embedding_model)
        embeddings_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
        embedding_name = args.embedding_model
    elif args.file_path and not args.embedding_model:
        print("Embedding model: sentence-transformers/all-mpnet-base-v2")
        if device == "cuda":
            with OnDevice(dtype="auto", device="meta"):
                embeddings_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2',offload_state_dict=True,output_attentions=True,output_hidden_states=True,low_cpu_mem_usage=False,offload_folder="/home/darf3/buddy/experiments/offload")
        else:
            embeddings_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        embeddings_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2', use_fast=False,return_tensors="pt",padding_side="left")
        embeddings, doc_text = gen_emb(args.file_path,(embeddings_tokenizer,embeddings_model), device=device)
        embedding_name = "sentence-transformers/all-mpnet-base-v2"
    elif args.embedding_model:
        print(f"Embedding model: {args.embedding_model}")
        embeddings, doc_text =gen_emb(None,args.embedding_model, device=device)
        embeddings_model = AutoModel.from_pretrained(args.embedding_model)
        embeddings_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)

   
  
    
    elif embeddings is None and args.embeddings_path and args.query:
        # Load the embeddings
        print(f"Embeddings about to be loaded: {args.embeddings_path}")
        embeddings = load_embeddings(args.embeddings_path)
        query = args.query
        embedding_name = args.embeddings_path
    
   


    elif args.query and embeddings is None and args.embeddings_path is None:
        # Load the default embeddings
        emb = input("Enter the path to the embeddings file or document: ")
        embeddings = load_embeddings(emb)
        query = args.query
        embedding_name = "default_embeddings.pt"
    elif args.embeddings_path and args.query is None:
        # Load the default query
        print("No query provided, provide a query")
        query = input("Enter a query: ")
    
    
    
    #load embeddings into a dataloader
    # if embeddings is not None:
    #     print("Embeddings loaded")
    #     embeddings = DataLoader(embeddings, batch_size=1, shuffle=False, num_workers=2)

    
    if embeddings_model is not None:
        print("Embeddings model loaded")
        print(f"before deepspeed: {(torch.cuda.memory_allocated()/1000)}, {(torch.cuda.memory_reserved()/1000)}")
        
    
        # accelerator.state.deepspeed_plugin.deepspeed_config[
        #     "train_micro_batch_size_per_gpu"] = 1
        # os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer.pad_token = tokenizer.eos_token
        embeddings_tokenizer.pad_token = embeddings_tokenizer.eos_token
        deepspeed.init_distributed("nccl")
        
        accelerator.register_for_checkpointing(embeddings_model)

        
        # embeddings_model.resize_token_embeddings(len(embeddings_tokenizer), pad_to_multiple_of=512)
        accelerator.prepare(embeddings_model, embeddings_tokenizer)
        embeddings_model = embeddings_model.to(device)
        embeddings_model = dispatch_model(embeddings_model,
            device_map=infer_auto_device_map(
                embeddings_model,
                dtype=torch.float16,
                max_memory={0: "7GiB", "cpu": "48GiB"},
            ),
        offload_dir="/home/darf3/buddy/experiments/offload",
        offload_buffers=True,)
        embeddings_model.eval()

        accelerator.prepare(model, tokenizer)
        model = model.to(device)
        model = dispatch_model(model,
            device_map=infer_auto_device_map(
                model,
                dtype=torch.float16,
                max_memory={0: "7GiB", "cpu": "48GiB"},
            ),
        offload_dir="/home/darf3/buddy/experiments/offload",
        offload_buffers=True,)
        accelerator.print(f"after deepspeed: {(torch.cuda.memory_allocated()/1000)}, {(torch.cuda.memory_reserved()/1000)}")



        
        
    else:
        embeddings_model = model
        embeddings_tokenizer = tokenizer
        embeddings_model = embeddings_model.to(device)
        embeddings_model.eval()
        print("Embeddings model not loaded, using same model as QA model")

    if args.input:
        # Run the model and get predictions
        predictions = run_model(args.input)
        
        # Print the predictions
        print(f"Predictions: {predictions}")
    else:
        while True:
            text = None
            if query == prev_query:
                query = None
                answered = False
            if embeddings is not None and query is None:
                print(f"Embeddings: {embedding_name}")
                query = input("Enter a query: ")
                answered = False
            elif embeddings is None and query is None:
                # Prompt the user for input
                text = input("Enter some text: ")

            if query and (text==None or text==""):
                text = query
            elif query==None and text:
                query = text
            elif query==None and text==None:
                continue

            
            # Check if the user wants to exit
            if "exit app" in text.lower():
                break 

            if embeddings is not None and answered == False and query != prev_query:
                print(f"Embeddings: {embedding_name}")
                if query == "" or query == None:
                    print("Query is empty")
                    query = input("Enter a query: ")
                print(f"Query: {query}")

                # Preprocess and tokenize the query
                # tokenized_query = preprocess_and_tokenize_query(query, embeddings_tokenizer
                with torch.no_grad():
                    tokenized_query = embeddings_tokenizer(
                        query,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)
                
                    model_output = embeddings_model(**tokenized_query)
                    tokenized_query = mean_pooling(model_output, tokenized_query['attention_mask'])
                #tokenized_query = bot_cortex.search.model(**tokenized_query).pooler_output #Why do this? Why not just use the tokenized_query as is? Because the model expects a tensor of shape (1, 512), not (1, 1, 512)
                tokenized_query = tokenized_query.detach()
                print(f"Tokenized query type: {type(tokenized_query)}")


                print(f"Working on answer...")
                # Run QA inference
                answer = run_qa_inference(model, tokenizer, doc_text, embeddings, tokenized_query, embeddings_model, embeddings_tokenizer, device=device)

                # Print the answer
                print(f"Answer to the question {query} based on the document {embedding_name}: {answer}")
                prev_query = query
                query = None
                answered = True
            elif embeddings is None:
                # Run the model and get predictions
                response = run_model(tokenizer, model, text, device=device)
                response = response.replace(text, "")
                # Print the predictions
                print(f"Response: {response}")