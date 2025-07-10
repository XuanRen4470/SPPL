import sys
import os
import torch
import numpy as np
# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from peft import PeftModel
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import re
import torch.nn.functional as F
from nltk.corpus import stopwords
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.function import HOME_DIRECTORY
from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity
import torch



def compute_similarity_scores(model, tokenizer, initial_prediction_list, predicted_outputs, device='cuda'):
    assert len(initial_prediction_list) == len(predicted_outputs), "The number of ground truths and predicted outputs must match."
    # ------------
    similarity_scores = 0
    num_pairs = len(initial_prediction_list)
    similarity_list = []
    layer_similarity_dict = defaultdict(list)
    for ground_truth, predicted_output in zip(initial_prediction_list, predicted_outputs):
        # Tokenize the ground truth and predicted output
        inputs_gt = tokenizer(ground_truth, return_tensors="pt")
        inputs_pred = tokenizer(predicted_output, return_tensors="pt")

        inputs_gt = inputs_gt.to(device)
        inputs_pred = inputs_pred.to(device)
        
        # Get embeddings for ground truth and predicted output
        with torch.no_grad():
            outputs_gt = model(**inputs_gt, output_hidden_states=True)
            outputs_pred = model(**inputs_pred, output_hidden_states=True)

        similarities = []
        num_layers = len(outputs_gt.hidden_states)
        for layer_num in range(num_layers):
            # Get embeddings for the current layer
            embedding_gt = outputs_gt.hidden_states[layer_num].mean(dim=1)
            embedding_pred = outputs_pred.hidden_states[layer_num].mean(dim=1)

            # Compute cosine similarity for the current layer
            similarity = F.cosine_similarity(embedding_gt, embedding_pred)
            similarities.append(similarity.item())
            layer_similarity_dict[layer_num].append(similarity.item())

        # Average the similarities across all layers
        avg_similarity = sum(similarities) / len(similarities)
        avg_similarity = 1 - avg_similarity
        similarity_scores += avg_similarity
        similarity_list.append(avg_similarity)
        
    similarity_scores /= num_pairs
    final_layer_similarity_dict = {layer: sum(values)/len(values) for layer, values in layer_similarity_dict.items()}

    for index in range(len(final_layer_similarity_dict)):
        value_item = float(1 - final_layer_similarity_dict[index])
        value_item = float(f"{value_item:.3g}")
        final_layer_similarity_dict[index] = value_item

    # final_layer_similarity_dict = float(1 - final_layer_similarity_dict)
    # final_layer_similarity_dict = float(f"{final_layer_similarity_dict:.3g}")
    return similarity_scores, final_layer_similarity_dict


import torch
import torch.nn.functional as F
from collections import defaultdict

def compute_last_layer_similarity_scores(model, tokenizer, groundtruth_list, predicted_outputs, device='cuda'):
    similarity_scores = 0
    similarity_list = []

    for ground_truth, predicted_output in zip(groundtruth_list[:len(predicted_outputs)], predicted_outputs):
        # Tokenize
        inputs_gt = tokenizer(ground_truth, return_tensors="pt").to(device)
        inputs_pred = tokenizer(predicted_output, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs_gt = model(**inputs_gt, output_hidden_states=True)
            outputs_pred = model(**inputs_pred, output_hidden_states=True)

        # Get last layer hidden states and average over tokens
        embedding_gt = outputs_gt.hidden_states[-1].mean(dim=1)
        embedding_pred = outputs_pred.hidden_states[-1].mean(dim=1)

        # Cosine similarity
        similarity = F.cosine_similarity(embedding_gt, embedding_pred).item()
        similarity_scores += similarity
        similarity_list.append(similarity)
    return similarity_list


# def compute_similarity_scores_question_attached(model, tokenizer, initial_prediction_list, predicted_outputs, data_list, model_name, device='cuda'):
#     questions = []
#     for problem_id, item in enumerate(data_list):
#         question = item['question']
#         original_question = item['original_question']

#         # Formatting the question based on the model type
#         if 'mistral' in model_name:
#             formatted_question = f"[INST] {question} [/INST]"
#             original_question = f"[INST] {original_question} [/INST]"
#         else:
#             formatted_question = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
        
#         formatted_question = formatted_question + ' '
#         questions.append(formatted_question)

# from collections import defaultdict
# import torch
# import torch.nn.functional as F

def compute_similarity_scores_question_attached(model, tokenizer, ground_truths, predictions, data_list, model_name, device='cuda'):
    questions = []
    for problem_id, item in enumerate(data_list):
        question = item['question']
        original_question = item['original_question']

        # Formatting the question based on the model type
        if 'mistral' in model_name:
            formatted_question = f"[INST] {question} [/INST]"
            original_question = f"[INST] {original_question} [/INST]"
        else:
            formatted_question = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
        
        formatted_question = formatted_question + ' '
        questions.append(formatted_question)

    """
    Compute similarity scores between the predicted outputs and ground truths, 
    with the question attached to both, but excluding the question's embedding during similarity calculation.

    Args:
        model: The model to use for computing embeddings.
        tokenizer: The tokenizer to use with the model.
        questions (list): List of questions.
        ground_truths (list): List of ground truth answers (without questions).
        predictions (list): List of predicted answers (without questions).
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        similarity_scores (float): Average similarity score across all pairs.
        final_layer_similarity_dict (dict): Layer-wise similarity scores.
    """
    assert len(questions) == len(ground_truths) == len(predictions), (
        "The number of questions, ground truths, and predictions must match."
    )
    
    similarity_scores = 0
    num_pairs = len(ground_truths)
    similarity_list = []
    layer_similarity_dict = defaultdict(list)

    for question, ground_truth, prediction in zip(questions, ground_truths, predictions):
        # Attach the question to the ground truth and prediction
        gt_with_question = f"{question} {ground_truth}"
        pred_with_question = f"{question} {prediction}"

        # Tokenize the combined sequences
        inputs_gt = tokenizer(gt_with_question, return_tensors="pt")
        inputs_pred = tokenizer(pred_with_question, return_tensors="pt")

        inputs_gt = inputs_gt.to(device)
        inputs_pred = inputs_pred.to(device)
        
        # Get embeddings for ground truth and predicted output
        with torch.no_grad():
            outputs_gt = model(**inputs_gt, output_hidden_states=True)
            outputs_pred = model(**inputs_pred, output_hidden_states=True)

        similarities = []
        num_layers = len(outputs_gt.hidden_states)
        for layer_num in range(num_layers):
            # Get embeddings for the current layer, excluding the question's embedding
            embedding_gt = outputs_gt.hidden_states[layer_num].mean(dim=1)
            embedding_pred = outputs_pred.hidden_states[layer_num].mean(dim=1)

            # Only compute the similarity of the prediction and ground truth embeddings
            similarity = F.cosine_similarity(embedding_gt, embedding_pred)
            similarities.append(similarity.item())
            layer_similarity_dict[layer_num].append(similarity.item())

        # Average the similarities across all layers
        avg_similarity = sum(similarities) / len(similarities)
        avg_similarity = 1 - avg_similarity  # Convert to dissimilarity
        similarity_scores += avg_similarity
        similarity_list.append(avg_similarity)
        
    similarity_scores /= num_pairs
    final_layer_similarity_dict = {layer: sum(values) / len(values) for layer, values in layer_similarity_dict.items()}

    # Format the final layer similarity values
    for index in range(len(final_layer_similarity_dict)):
        value_item = float(1 - final_layer_similarity_dict[index])
        value_item = float(f"{value_item:.3g}")
        final_layer_similarity_dict[index] = value_item

    return similarity_scores, final_layer_similarity_dict


def compute_complexity_scores(model, tokenizer, predicted_outputs, hidden_states_layer_num, device='cuda'):
    similarity_scores = 0
    embedding_pred_total_list = []
    similarity_list = []
    embedding_list = []

    def compute_avg_embeddings(predicted_outputs, tokenizer, model, device, hidden_states_layer_num):
        """
        Compute the average embeddings for the given predicted outputs.

        Args:
            predicted_outputs (list): A list of predicted output strings.
            tokenizer: The tokenizer to tokenize the predicted outputs.
            model: The model to obtain hidden states from.
            device: The device to run the computations on.
            hidden_states_layer_num (int): The layer number to compute embeddings for.
                Set to -1 to compute embeddings for all layers.

        Returns:
            If hidden_states_layer_num == -1:
                A list of tensors, each tensor is the average embedding for that layer across all examples.
                The list length is equal to the number of layers in the model.
            Else:
                A single tensor representing the average embedding of the specified layer across all examples.
        """
        if hidden_states_layer_num == -1:
            # Collect embeddings for all layers
            num_layers = None  # Will be set after processing the first example
            embeddings_per_layer = None  # Will be initialized after processing the first example

            for predicted_output in predicted_outputs:
                # Tokenize the predicted output
                inputs_pred = tokenizer(predicted_output, return_tensors="pt").to(device)

                # Get model outputs with hidden states
                with torch.no_grad():
                    outputs_pred = model(**inputs_pred, output_hidden_states=True)

                # Initialize num_layers and embeddings_per_layer
                if num_layers is None:
                    num_layers = len(outputs_pred.hidden_states)
                    embeddings_per_layer = [[] for _ in range(num_layers)]  # List of lists

                # Collect mean embeddings for each layer
                for layer_num in range(num_layers):
                    # Hidden state shape: [batch_size, seq_len, hidden_size]
                    # We average over the sequence length (tokens)
                    embedding_pred = outputs_pred.hidden_states[layer_num].mean(dim=1)  # Shape: [batch_size, hidden_size]
                    embeddings_per_layer[layer_num].append(embedding_pred.squeeze(0))   # Shape: [hidden_size]

            # Compute average embeddings for each layer
            avg_embeddings = []
            for layer_num in range(num_layers):
                # Stack embeddings to form a tensor of shape [num_examples, hidden_size]
                layer_embeddings = torch.stack(embeddings_per_layer[layer_num])  # Shape: [num_examples, hidden_size]
                # Compute mean over all examples
                avg_embedding = layer_embeddings.mean(dim=0)  # Shape: [hidden_size]
                avg_embeddings.append(avg_embedding)

            # The output is a list of average embeddings for each layer
            return avg_embeddings  # List length: num_layers, each element shape: [hidden_size]

        else:
            # Collect embeddings for the specified layer only
            embeddings = []

            for predicted_output in predicted_outputs:
                # Tokenize the predicted output
                inputs_pred = tokenizer(predicted_output, return_tensors="pt").to(device)

                # Get model outputs with hidden states
                with torch.no_grad():
                    outputs_pred = model(**inputs_pred, output_hidden_states=True)

                # Get the embedding for the specified layer
                embedding_pred = outputs_pred.hidden_states[hidden_states_layer_num].mean(dim=1)  # Shape: [batch_size, hidden_size]
                embeddings.append(embedding_pred.squeeze(0))  # Shape: [hidden_size]

            # Compute the average embedding across all examples
            embeddings_tensor = torch.stack(embeddings)  # Shape: [num_examples, hidden_size]
            avg_embedding = embeddings_tensor.mean(dim=0)  # Shape: [hidden_size]

            # The output is a single average embedding for the specified layer
            return avg_embedding  # Shape: [hidden_size]
    
    avg_embeddings = compute_avg_embeddings(predicted_outputs, tokenizer, model, device, hidden_states_layer_num)

    for i, predicted_output in enumerate(predicted_outputs):
        # Tokenize the ground truth and predicted output
        inputs_pred = tokenizer(predicted_output, return_tensors="pt")
        inputs_pred = inputs_pred.to(device)

        # Get embeddings for ground truth and predicted output
        with torch.no_grad():
            outputs_pred = model(**inputs_pred, output_hidden_states=True)

        
        if hidden_states_layer_num == -1:
            similarities = []
            num_layers = len(outputs_pred.hidden_states)
            for layer_num in range(num_layers):
                # Get embeddings for the current layer
                embedding_pred = outputs_pred.hidden_states[layer_num].mean(dim=1)
                avg_embeddings_item = avg_embeddings[layer_num]
                # Compute cosine similarity for the current layer
                similarity = F.cosine_similarity(avg_embeddings_item, embedding_pred)
                similarities.append(similarity.item())

                # Optionally collect embeddings
                embedding_pred_total_list.append(embedding_pred)

            # Average the similarities across all layers
            avg_similarity = sum(similarities) / len(similarities)
            similarity_scores += avg_similarity
            similarity_list.append(avg_similarity)

        else:
            # Compute embeddings and similarity for the specified layer
            embedding_pred = outputs_pred.hidden_states[hidden_states_layer_num].mean(dim=1)
            avg_embeddings_item = avg_embeddings
            similarity = F.cosine_similarity(avg_embeddings_item, embedding_pred)

            embedding_pred_total_list.append(embedding_pred)
            similarity_scores += similarity.item()
            similarity_list.append(similarity.item())
        
# ---------------
    complexity_list = torch.cat(embedding_pred_total_list, dim=0)  # Shape: [num_pairs, embedding_dim]
    # Compute the mean embedding across all pairs
    complexity_list = complexity_list.mean(dim=0)

    avg_complexity_score = (similarity_scores / len(predicted_outputs))
    return avg_complexity_score 


import heapq

class LimitedQueue:
    def __init__(self, max_size=10):
        """
        使用一个列表 self._heap 来存储元素，元素以 (-value, key, value) 的形式存入。
        之所以存 -value，是因为 Python 的 heapq 是最小堆，
        这样就可以把原本的 'value' 的最大值映射为最小的 -value，来达到最大堆的效果。
        """
        self.max_size = max_size
        self._heap = []

    def push(self, key, value):
        """
        当新元素的 value 小于当前队列中最大的 value 时，移除最大的，并插入新元素；
        否则什么都不做。
        """
        if len(self._heap) < self.max_size:
            # 若还未达到最大容量，直接压入堆
            heapq.heappush(self._heap, (-value, key, value))
        else:
            # 当前队列已满，先看看最大的那个值是多少
            # 因为存的是 -value，所以最大的原始 value 其实是 -self._heap[0][0]
            current_largest_value = -self._heap[0][0]

            if value < current_largest_value:
                # 新元素的 value 更小，弹出最大的那个，然后插入新元素
                heapq.heapreplace(self._heap, (-value, key, value))
            # 如果新元素不更小，则什么也不做（忽略该元素）

    def get_all(self):
        """
        返回当前队列中所有元素，以 (key, value) 形式。
        注意，这里顺序并不一定是按 value 排序的，因为内部是堆结构。
        """
        return [(item[1], item[2]) for item in self._heap]

    def get_current_largest(self):
        """
        获取当前队列中最大的 value 以及对应 key。
        如果队列为空，返回 None。
        """
        if not self._heap:
            return None
        largest_value = -self._heap[0][0]
        largest_key = self._heap[0][1]
        return (largest_key, largest_value)
    
    def get_current_top_n(self, top_n):
        """
        返回当前队列中 value 最小的 top_n 个元素的 key 组成的列表。
        如果 top_n 大于队列长度，则返回所有元素的 key。
        """
        if not self._heap:
            return []
        
        # 从堆中提取所有元素并按 value 由小到大排序
        sorted_elements = sorted(self._heap, key=lambda x: -x[0])  # -x[0] 是因为存储的是 -value
        # 取前 top_n 个元素的 key
        return [item[1] for item in sorted_elements[:top_n]]

    def get_size(self):
        """
        返回当前队列内的元素数量
        """
        return len(self._heap)

def find_similar_examples(model, tokenizer, initial_prediction_list, device='cuda', n_similar_self_generated_examples = 10, top_n = 2):
    
    num_pairs = len(initial_prediction_list)
    layer_similarity_dict = defaultdict(list)
    similarity_queue = LimitedQueue(max_size=n_similar_self_generated_examples)
    for i, current_predict_item in enumerate(initial_prediction_list):
        similarity_scores = 0
        for item_compared_with in initial_prediction_list:
            # Tokenize the ground truth and predicted output
            inputs_1 = tokenizer(current_predict_item, return_tensors="pt")
            inputs_2 = tokenizer(item_compared_with, return_tensors="pt")

            inputs_1 = inputs_1.to(device)
            inputs_2 = inputs_2.to(device)
            
            # Get embeddings for ground truth and predicted output
            with torch.no_grad():
                outputs_1 = model(**inputs_1, output_hidden_states=True)
                outputs_2 = model(**inputs_2, output_hidden_states=True)

            similarities = []
            num_layers = len(outputs_1.hidden_states)
            for layer_num in range(num_layers):
                # Get embeddings for the current layer
                embedding_1 = outputs_1.hidden_states[layer_num].mean(dim=1)
                embedding_2 = outputs_2.hidden_states[layer_num].mean(dim=1)

                # Compute cosine similarity for the current layer
                similarity = F.cosine_similarity(embedding_1, embedding_2)
                similarities.append(similarity.item())
                layer_similarity_dict[layer_num].append(similarity.item())

            # Average the similarities across all layers
            avg_similarity = sum(similarities) / len(similarities)
            avg_similarity = 1 - avg_similarity
            similarity_scores += avg_similarity
        
        similarity_scores /= num_pairs
        similarity_queue.push(i, similarity_scores)

        print("队列中的所有元素 (顺序无特别含义):")
        print(similarity_queue.get_all())

        print("当前队列中最大的元素是:", similarity_queue.get_current_largest())
        
    similar_example_index_list = similarity_queue.get_current_top_n(top_n)
    print(f"队列中 value 最小的前 {top_n} 个元素的 key:")
    print(similarity_queue.get_current_top_n(top_n))

    selected_example_list = []
    for index in similar_example_index_list:
        selected_example_list.append(initial_prediction_list[index])
    return similar_example_index_list, selected_example_list



def extract_embeddings(embedding_list):
    # 将嵌入列表转换为 NumPy 数组
    embeddings_array = np.vstack([embedding.cpu().numpy() for embedding in embedding_list])
    return embeddings_array

def visualize_with_tsne(embeddings_array, data_name, hidden_states_layer_num, save_dir = '', x_lim=None, y_lim=None):
    # 使用 t-SNE 将嵌入向量降维到二维
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_array)
    
    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='green', alpha=0.5)
    plt.title(f't-SNE {data_name} layer {hidden_states_layer_num}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    # plt.show()

    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

    save_path = os.path.join(save_dir, f"{data_name}_layer_{hidden_states_layer_num}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE 可视化图已保存到 {save_path}")

def visualize_with_tsne(embeddings_array, data_name, hidden_states_layer_num, save_dir='', x_lim=None, y_lim=None):
    import os
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # 检查数据是否有效
    print(f"embeddings_array shape: {embeddings_array.shape}")
    if embeddings_array.shape[0] == 0:
        raise ValueError("Embedding array is empty.")

    # 动态调整 perplexity
    n_samples = embeddings_array.shape[0]
    perplexity = max(1, min(30, n_samples // 3))
    print(f"Using perplexity: {perplexity}")

    # t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_array)
    print(f"t-SNE reduced dimensions range: x=[{embeddings_2d[:, 0].min()}, {embeddings_2d[:, 0].max()}], "
          f"y=[{embeddings_2d[:, 1].min()}, {embeddings_2d[:, 1].max()}]")

    # 绘制 t-SNE 图
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='green', alpha=0.5)
    plt.title(f't-SNE {data_name} layer {hidden_states_layer_num}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # 自动动态设置 xlim 和 ylim
    x_min, x_max = embeddings_2d[:, 0].min() - 5, embeddings_2d[:, 0].max() + 5  # 增加边距
    y_min, y_max = embeddings_2d[:, 1].min() - 5, embeddings_2d[:, 1].max() + 5  # 增加边距
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # 保存图像
    save_path = os.path.join(save_dir, f"{data_name}_layer_{hidden_states_layer_num}.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.show()  # 显示图像，确认结果
    print(f"t-SNE 可视化图已保存到 {save_path}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def visualize_tsne_total(embeddings_array, categories, category_names, data_name, save_dir='', x_lim=None, y_lim=None):
    """
    使用 t-SNE 降维并在一张图中可视化不同类别的数据点

    参数:
        embeddings_array: numpy.ndarray, 数据嵌入 (n_samples, n_features)
        categories: list 或 numpy.ndarray, 每个点的类别标签 (n_samples,)
        category_names: list, 每个类别的名称
        data_name: str, 数据集名称
        save_dir: str, 保存路径
        x_lim: tuple, 可选，x 轴范围
        y_lim: tuple, 可选，y 轴范围
    """
    # 检查输入
    if len(embeddings_array) != len(categories):
        raise ValueError("Embedding array and categories must have the same length.")

    # t-SNE 降维
    perplexity = max(1, min(30, embeddings_array.shape[0] // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_array)

    # 创建颜色映射
    unique_categories = np.unique(categories)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))  # 自动分配颜色

    # 绘制 t-SNE 图
    plt.figure(figsize=(10, 8))
    for i, cat in enumerate(unique_categories):
        mask = (categories == cat)
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=category_names[cat],
            alpha=0.6,
            s=50
        )
    
    # 添加图例和标题
    plt.legend(title="Categories", loc='best')
    plt.title(f't-SNE Visualization for {data_name}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # 设置坐标范围
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)

    # 保存图像
    save_path = os.path.join(save_dir, f"{data_name}_tsne.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.show()  # 显示图像
    print(f"t-SNE 可视化图已保存到 {save_path}")



def visualize_with_pca(embeddings_array, data_name, hidden_states_layer_num, save_dir = '', x_lim=None, y_lim=None):
    # 使用 PCA 将嵌入向量降维到二维
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_array)
    
    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', alpha=0.5)
    plt.title(f'PCA {data_name} layer {hidden_states_layer_num}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    # plt.show()

    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

    save_path = os.path.join(save_dir, f"{data_name}_layer_{hidden_states_layer_num}.png")
    plt.savefig(save_path)
    plt.close()
    # print(f"PCA 可视化图已保存到 {save_path}")

def calc_total_diversity(embedding_list):
    # 将嵌入列表转换为 NumPy 数组
    embeddings_array = np.vstack([embedding.cpu().numpy() for embedding in embedding_list])
    
    # 计算平均向量
    mean_embedding = np.mean(embeddings_array, axis=0)
    
    # 计算每个嵌入向量与平均向量之间的欧氏距离
    distances = np.linalg.norm(embeddings_array - mean_embedding, axis=1)
    
    # 计算离散程度（标准差和方差）
    std_deviation = np.std(distances)
    # variance = np.var(distances)
    
    # print("离散程度（标准差）:", std_deviation)
    # print("离散程度（方差）:", variance)
    return std_deviation
    # return variance

def compute_diversity_pca_scores(model, tokenizer, predicted_outputs, hidden_states_layer_num, train_task_name, data_name, device='cuda'):
    embedding_list = []
    
    for predicted_output in predicted_outputs:
        # 对预测输出进行标记化
        inputs_pred = tokenizer(predicted_output, return_tensors="pt")
        inputs_pred = inputs_pred.to(device)
    
        # 获取模型的隐藏状态
        with torch.no_grad():
            outputs_pred = model(**inputs_pred, output_hidden_states=True)
    
        # 获取指定层的嵌入向量，并在时间维度上取平均
        embedding_pred = outputs_pred.hidden_states[hidden_states_layer_num].mean(dim=1)
        embedding_list.append(embedding_pred)

    embeddings_array = extract_embeddings(embedding_list)
    
    pca_save_path = f"{HOME_DIRECTORY}/perplexity_record/data_distribution_pca/{train_task_name.lower()}/{hidden_states_layer_num}"
    # tsne_save_path = f"{HOME_DIRECTORY}/perplexity_record/data_distribution_tsne/{train_task_name.lower()}/{hidden_states_layer_num}"

    os.makedirs(pca_save_path, exist_ok=True)
    
    # 可视化嵌入向量分布
    visualize_with_pca(embeddings_array, data_name, hidden_states_layer_num, save_dir = pca_save_path, x_lim=(-3, 1), y_lim=(-1, 1))
    # visualize_with_tsne(embeddings_array, data_name, hidden_states_layer_num, save_dir = tsne_save_path)

    

    # 计算离散程度，不需要再除以样本数量
    variance_scores = calc_total_diversity(embedding_list)
    
    return variance_scores


def compute_diversity_tsne_scores(model, tokenizer, predicted_outputs, hidden_states_layer_num, train_task_name, data_name, gpt4_prediction_list, device='cuda'):
    embedding_list = []
    predicted_outputs = predicted_outputs + gpt4_prediction_list
    for predicted_output in predicted_outputs:
        # 对预测输出进行标记化
        inputs_pred = tokenizer(predicted_output, return_tensors="pt")
        inputs_pred = inputs_pred.to(device)
    
        # 获取模型的隐藏状态
        with torch.no_grad():
            outputs_pred = model(**inputs_pred, output_hidden_states=True)
    
        # 获取指定层的嵌入向量，并在时间维度上取平均
        embedding_pred = outputs_pred.hidden_states[hidden_states_layer_num].mean(dim=1)
        embedding_list.append(embedding_pred)

    embeddings_array = extract_embeddings(embedding_list)
    
    tsne_save_path = f"{HOME_DIRECTORY}/perplexity_record/data_distribution_tsne/{train_task_name.lower()}/{hidden_states_layer_num}"

    os.makedirs(tsne_save_path, exist_ok=True)
    
    # 可视化嵌入向量分布
    visualize_with_tsne(embeddings_array, data_name, hidden_states_layer_num, save_dir = tsne_save_path, x_lim=(-3, 1), y_lim=(-1, 1))
    # visualize_tsne_total(embeddings_array, categories, category_names, data_name, save_dir='', x_lim=None, y_lim=None):


    # 计算离散程度，不需要再除以样本数量
    # variance_scores = calc_total_diversity(embedding_list)
    variance_scores = 0
    return variance_scores

def compute_similarity_scores_using_probabilities(model, tokenizer, ground_truths, predicted_outputs, device='cuda'):
    assert len(ground_truths) == len(predicted_outputs), "The number of ground truths and predicted outputs must match."
    similarity_scores = 0
    num_pairs = len(ground_truths)
    probs_gt_total_list = []
    probs_pred_total_list = []

    for ground_truth, predicted_output in zip(ground_truths, predicted_outputs):
        # Tokenize the ground truth and predicted output
        inputs_gt = tokenizer(ground_truth, return_tensors="pt").to(device)
        inputs_pred = tokenizer(predicted_output, return_tensors="pt").to(device)

        # Get logits for ground truth and predicted output
        with torch.no_grad():
            outputs_gt = model(**inputs_gt)
            outputs_pred = model(**inputs_pred)
        
        # Get probabilities by applying softmax to the logits
        probs_gt = torch.softmax(outputs_gt.logits, dim=-1)  # Shape: [1, seq_len_gt, vocab_size]
        probs_pred = torch.softmax(outputs_pred.logits, dim=-1)  # Shape: [1, seq_len_pred, vocab_size]

        # Average probabilities over the sequence length
        probs_gt_mean = probs_gt.mean(dim=1)  # Shape: [1, vocab_size]
        probs_pred_mean = probs_pred.mean(dim=1)  # Shape: [1, vocab_size]

        # Compute cosine similarity between the probability distributions
        similarity = F.cosine_similarity(probs_gt_mean, probs_pred_mean, dim=-1)

        probs_gt_total_list.append(probs_gt_mean)
        probs_pred_total_list.append(probs_pred_mean)

        similarity_scores += similarity.item()

    # Concatenate and average the probabilities over all pairs
    probs_gt_total = torch.cat(probs_gt_total_list, dim=0)  # Shape: [num_pairs, vocab_size]
    probs_pred_total = torch.cat(probs_pred_total_list, dim=0)  # Shape: [num_pairs, vocab_size]

    # Compute the mean probability distribution across all pairs
    probs_gt_total_mean = probs_gt_total.mean(dim=0)  # Shape: [vocab_size]
    probs_pred_total_mean = probs_pred_total.mean(dim=0)  # Shape: [vocab_size]

    # Compute cosine similarity between the mean probability distributions
    avg_total_similarity = F.cosine_similarity(probs_gt_total_mean.unsqueeze(0), probs_pred_total_mean.unsqueeze(0), dim=-1)

    similarity_scores /= num_pairs
    return similarity_scores, avg_total_similarity

def get_mean_embedding_without_stopwords(model, tokenizer, input_text, hidden_states_layer_num, stop_words, device='cuda'):
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_ids = inputs['input_ids'][0]  # Shape: [seq_length]

    # Convert input_ids to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    embeddings = outputs.hidden_states[hidden_states_layer_num][0]  # Shape: [seq_length, embedding_dim]

    # Build mask for non-stop words
    mask = []
    for token in tokens:
        # Remove special tokens
        if token in tokenizer.all_special_tokens:
            mask.append(False)
            continue
        # Remove subword prefixes (e.g., '##' in BERT)
        token_stripped = token.lstrip('##').lower()
        # Check if token is a stop word
        if token_stripped in stop_words:
            mask.append(False)
        else:
            mask.append(True)
    mask = torch.tensor(mask, dtype=torch.bool, device=device)

    # Apply mask to embeddings
    embeddings = embeddings[mask, :]
    
    # Compute mean embedding
    if embeddings.size(0) > 0:
        mean_embedding = embeddings.mean(dim=0)
    else:
        # Handle cases where all tokens are stop words
        mean_embedding = torch.zeros(embeddings.size(1), device=device)
    return mean_embedding

def compute_similarity_scores_without_stopwords(model, tokenizer, ground_truths, predicted_outputs, hidden_states_layer_num, device='cuda'):
    assert len(ground_truths) == len(predicted_outputs), "The number of ground truths and predicted outputs must match."

    similarity_scores = 0
    num_pairs = len(ground_truths)
    embedding_gt_total_list = []
    embedding_pred_total_list = []

    # Download and set up stop words
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

    for ground_truth, predicted_output in zip(ground_truths, predicted_outputs):
        # Get mean embeddings for ground truth and predicted output, excluding stop words
        embedding_gt = get_mean_embedding_without_stopwords(model, tokenizer, ground_truth, hidden_states_layer_num, stop_words, device)
        embedding_pred = get_mean_embedding_without_stopwords(model, tokenizer, predicted_output, hidden_states_layer_num, stop_words, device)

        # Compute cosine similarity
        similarity = F.cosine_similarity(embedding_gt.unsqueeze(0), embedding_pred.unsqueeze(0))
        similarity_scores += similarity.item()

        embedding_gt_total_list.append(embedding_gt)
        embedding_pred_total_list.append(embedding_pred)

    # Stack embeddings and compute the mean embeddings
    embedding_gt_total = torch.stack(embedding_gt_total_list)
    embedding_pred_total = torch.stack(embedding_pred_total_list)

    # Compute the mean embedding across all pairs
    embedding_gt_mean = embedding_gt_total.mean(dim=0)
    embedding_pred_mean = embedding_pred_total.mean(dim=0)

    avg_total_similarity = F.cosine_similarity(embedding_gt_mean.unsqueeze(0), embedding_pred_mean.unsqueeze(0))

    similarity_scores /= num_pairs
    return similarity_scores, avg_total_similarity.item()


def compute_similarity_scores_sentence_bert(model, ground_truths, predicted_outputs, device='cuda'):
    assert len(ground_truths) == len(predicted_outputs), "The number of ground truths and predicted outputs must match."

    similarity_scores = 0
    num_pairs = len(ground_truths)
    embedding_gt_total_list = []
    embedding_pred_total_list = []

    for ground_truth, predicted_output in zip(ground_truths, predicted_outputs):
        # Get embeddings for ground truth and predicted output using SentenceBERT
        embedding_gt = model.encode(ground_truth, convert_to_tensor=True, device=device)
        embedding_pred = model.encode(predicted_output, convert_to_tensor=True, device=device)

        # Compute cosine similarity between the ground truth and predicted output
        similarity = F.cosine_similarity(embedding_gt, embedding_pred, dim=0)

        embedding_gt_total_list.append(embedding_gt)
        embedding_pred_total_list.append(embedding_pred)

        similarity_scores += similarity.item()

    # Stack embeddings
    embedding_gt_total = torch.stack(embedding_gt_total_list, dim=0)  # Shape: [num_pairs, embedding_dim]
    embedding_pred_total = torch.stack(embedding_pred_total_list, dim=0)  # Shape: [num_pairs, embedding_dim]

    # Compute the mean embedding across all pairs
    embedding_gt_mean = embedding_gt_total.mean(dim=0)  # Shape: [embedding_dim]
    embedding_pred_mean = embedding_pred_total.mean(dim=0)

    # Compute similarity between mean embeddings
    avg_total_similarity = F.cosine_similarity(embedding_gt_mean, embedding_pred_mean, dim=0)

    similarity_scores /= num_pairs
    return similarity_scores, avg_total_similarity.item()

def compute_similarity_scores_sentence_bert_individual_sentence(model, ground_truths, predicted_outputs, device='cuda'):
    assert len(ground_truths) == len(predicted_outputs), "The number of ground truths and predicted outputs must match."

    similarity_scores = 0
    num_pairs = 0
    embedding_gt_total_list = []
    embedding_pred_total_list = []

    for ground_truth, predicted_output in zip(ground_truths, predicted_outputs):
        embedding_gt_total = model.encode(ground_truth, convert_to_tensor=True, device=device)
        embedding_pred_total = model.encode(predicted_output, convert_to_tensor=True, device=device)
        embedding_gt_total_list.append(embedding_gt_total)
        embedding_pred_total_list.append(embedding_pred_total)
        # Get embeddings for ground truth and predicted output using SentenceBERT

        

        ground_truth_paragraphs = ground_truth.split('\n\n')
        predicted_output_paragraphs = predicted_output.split('\n\n')

        # Step 2: Split each paragraph into sentences
        ground_truth_list = []
        predicted_output_list = []
        sentence_split_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'

        for paragraph in ground_truth_paragraphs:
            sentences = re.split(sentence_split_pattern, paragraph)
            ground_truth_list.extend(sentences)

        for paragraph in predicted_output_paragraphs:
            sentences = re.split(sentence_split_pattern, paragraph)
            predicted_output_list.extend(sentences)
        


        predicted_output_embedding_list = []
        ground_truth_embedding_list = []
        for ground_truth in ground_truth_list:
            embedding_gt = model.encode(ground_truth, convert_to_tensor=True, device=device)
            ground_truth_embedding_list.append(embedding_gt)
        
        for embedding_pred in predicted_output_list:
            embedding_pred = model.encode(predicted_output, convert_to_tensor=True, device=device)
            predicted_output_embedding_list.append(embedding_pred)
        
        similarity_score_item = 0
        for embedding_gt in ground_truth_embedding_list:
            highest_similarity = 0
            for embedding_pred in predicted_output_embedding_list:
                # Compute cosine similarity between the ground truth and predicted output
                similarity = F.cosine_similarity(embedding_gt, embedding_pred, dim=0)
                if similarity > highest_similarity:
                    highest_similarity = similarity
            similarity_score_item += highest_similarity.item()
        similarity_score_item /= len(ground_truth_embedding_list)
        similarity_scores += similarity_score_item

    # Stack embeddings
    embedding_gt_total = torch.stack(embedding_gt_total_list, dim=0)  # Shape: [num_pairs, embedding_dim]
    embedding_pred_total = torch.stack(embedding_pred_total_list, dim=0)  # Shape: [num_pairs, embedding_dim]

    # Compute the mean embedding across all pairs
    embedding_gt_mean = embedding_gt_total.mean(dim=0)  # Shape: [embedding_dim]
    embedding_pred_mean = embedding_pred_total.mean(dim=0)

    # Compute similarity between mean embeddings
    avg_total_similarity = F.cosine_similarity(embedding_gt_mean, embedding_pred_mean, dim=0)

    similarity_scores /= len(ground_truths)
    return similarity_scores, avg_total_similarity.item()



def perplexity_calculation(data_list, model, tokenizer, train_task_name, model_name, similarity_compare_to_irrelevant_prediction, device='cuda', not_cap_perplexity = True):
    token_len_list = []
    perplexity_list = []
    IDF_list = []
    loss_list = []
    answer_char_count_total = 0
    for problem_id, item in enumerate(data_list):
        question = item['question']
        original_question = item['original_question']
        answer = item['answer']
        answer_char_count_total += len(answer)
        if similarity_compare_to_irrelevant_prediction:
#                     answer = """The Unanimous Declaration of the Thirteen United States of America . When, in the course of human events, it becomes necessary for one people to dissolve the political bonds which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the laws of nature and of nature\''s God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation 

# We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable rights, that among these are life, liberty and the pursuit of happiness. That to secure these rights, governments are instituted among men, deriving their just powers from the consent of the governed. That whenever any form of government becomes destructive to these ends, it is the right of the people to alter or to abolish it, and to institute new government, laying its foundation on such principles and organizing its powers in such form, as to them shall seem most likely to effect their safety and happiness.

# Prudence, indeed, will dictate that governments long established should not be changed for light and transient causes; and accordingly all experience hath shown that mankind are more disposed to suffer, while evils are sufferable, than to right themselves by abolishing the forms to which they are accustomed. But when a long train of abuses and usurpations, pursuing invariably the same object evinces a design to reduce them under absolute despotism, it is their right, it is their duty, to throw off such government, and to provide new guards for their future security. -- Such has been the patient sufferance of these colonies; and such is now the necessity which constrains them to alter their former systems of government. The history of the present King of Great Britain is a history of repeated injuries and usurpations, all having in direct object the establishment of an absolute tyranny over these states. To prove this, let facts be submitted to a candid world. """
            # answer = dataset_list[0][1][0]['answer']
            # answer = irrelevant_prediction
            a = 1

        # Formatting the question based on the model type
        if 'mistral' in model_name:
            formatted_question = f"[INST] {question} [/INST]"
            original_question = f"[INST] {original_question} [/INST]"
        else:
            formatted_question = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
        
        q_and_a = formatted_question + ' ' + answer
        answer_inputs = tokenizer(answer, return_tensors="pt")
        answer_input_ids = answer_inputs['input_ids']
        answer_input_ids = answer_input_ids.to(device)

        # Decoding each token ID into its corresponding token
        tokens = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in answer_input_ids[0]]
        token_len = len(tokens)
        token_len_list.append(token_len)
        # print(tokens)

        q_and_a_inputs = tokenizer(q_and_a, return_tensors="pt")
        q_a_input_ids = q_and_a_inputs["input_ids"].to(device)
        question_inputs = tokenizer(formatted_question, return_tensors="pt")
        question_len = question_inputs["input_ids"].shape[1]
        labels = q_a_input_ids.clone()
        labels[:, :question_len] = -100

        with torch.no_grad():
            outputs = model(input_ids=q_a_input_ids, labels=labels)
            loss = outputs.loss
            perplexity = torch.exp(loss)
            loss_list.append(loss.item())

            # Forward pass: answer only (to get perplexity of answer alone)
            # Here, we compute loss on answer_input_ids starting from the second token
            
            if len(answer_input_ids[0]) > 1:
                outputs_y_only = model(input_ids=answer_input_ids, labels=answer_input_ids)
                y_only_loss = outputs_y_only.loss
                y_only_perplexity = torch.exp(y_only_loss)

                # Compute IDF
                IDF = perplexity / y_only_perplexity
                IDF_list.append(IDF.item())

            if perplexity > 30:
                # a = 1
                if not_cap_perplexity:
                    perplexity_list.append(perplexity.item())
            else:
                perplexity_list.append(perplexity.item())
            
            if perplexity_list == []:
                perplexity_list.append(1000000)

    average_perplexity = sum(perplexity_list) / len(perplexity_list) if perplexity_list else float('inf')
    average_IDF = sum(IDF_list) / len(IDF_list) if IDF_list else float('inf')
    average_loss = sum(loss_list) / len(loss_list) if loss_list else float('inf')
    average_token_len = sum(token_len_list) / len(token_len_list)
    average_char_len = answer_char_count_total / len(data_list)

    return average_perplexity, average_IDF, average_loss, average_token_len, average_char_len



def in_context_perplexity_calculation(data_list, model, tokenizer, train_task_name, model_name, similarity_compare_to_irrelevant_prediction, device='cuda', not_cap_perplexity = True):
    token_len_list = []
    perplexity_list = []
    IDF_list = []
    loss_list = []
    answer_char_count_total = 0
    for problem_id, item in enumerate(data_list):
        question = item['question']
        original_question = item['original_question']
        answer = item['answer']
        answer_char_count_total += len(answer)

        # Formatting the question based on the model type
        if 'mistral' in model_name:
            formatted_question = f"[INST] {question} [/INST]"
            original_question = f"[INST] {original_question} [/INST]"
        else:
            formatted_question = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
        
        q_and_a = formatted_question + ' ' + answer
        answer_inputs = tokenizer(answer, return_tensors="pt")
        answer_input_ids = answer_inputs['input_ids']
        answer_input_ids = answer_input_ids.to(device)

        # Decoding each token ID into its corresponding token
        tokens = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in answer_input_ids[0]]
        token_len = len(tokens)
        token_len_list.append(token_len)
        # print(tokens)

        q_and_a_inputs = tokenizer(q_and_a, return_tensors="pt")
        q_a_input_ids = q_and_a_inputs["input_ids"].to(device)
        question_inputs = tokenizer(formatted_question, return_tensors="pt")
        question_len = question_inputs["input_ids"].shape[1]
        labels = q_a_input_ids.clone()
        labels[:, :question_len] = -100

        with torch.no_grad():
            outputs = model(input_ids=q_a_input_ids, labels=labels)
            loss = outputs.loss
            perplexity = torch.exp(loss)
            loss_list.append(loss.item())

            # Forward pass: answer only (to get perplexity of answer alone)
            # Here, we compute loss on answer_input_ids starting from the second token
            
            if len(answer_input_ids[0]) > 1:
                outputs_y_only = model(input_ids=answer_input_ids, labels=answer_input_ids)
                y_only_loss = outputs_y_only.loss
                y_only_perplexity = torch.exp(y_only_loss)

                # Compute IDF
                IDF = perplexity / y_only_perplexity
                IDF_list.append(IDF.item())

            if perplexity > 30:
                # a = 1
                if not_cap_perplexity:
                    perplexity_list.append(perplexity.item())
            else:
                perplexity_list.append(perplexity.item())
            
            if perplexity_list == []:
                perplexity_list.append(1000000)

    average_perplexity = sum(perplexity_list) / len(perplexity_list) if perplexity_list else float('inf')
    return average_perplexity

def calculate_perplexity(data_list, model, tokenizer, model_name, device='cuda'):
    perplexity_list = []
    loss_list = []
    answer_char_count_total = 0
    for problem_id, item in enumerate(data_list):
        question = item['question']
        original_question = item['original_question']
        answer = item['answer']
        answer_char_count_total += len(answer)

        # Formatting the question based on the model type
        if 'mistral' in model_name:
            formatted_question = f"[INST] {question} [/INST]"
            original_question = f"[INST] {original_question} [/INST]"
        else:
            formatted_question = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
        
        q_and_a = formatted_question + ' ' + answer
        q_and_a_inputs = tokenizer(q_and_a, return_tensors="pt")
        q_a_input_ids = q_and_a_inputs["input_ids"].to(device)
        question_inputs = tokenizer(formatted_question, return_tensors="pt")
        question_len = question_inputs["input_ids"].shape[1]
        labels = q_a_input_ids.clone()
        labels[:, :question_len] = -100

        with torch.no_grad():
            outputs = model(input_ids=q_a_input_ids, labels=labels)
            loss = outputs.loss
            perplexity = torch.exp(loss)
            loss_list.append(loss.item())
            perplexity_list.append(perplexity.item())
    return perplexity_list


# def in_context_perplexity_calculation_length_cutoff(data_list, model, tokenizer, model_name, length_cutoff, initial_prediction_list_total, device='cuda'):
#     max_cccc = len(initial_prediction_list_total['initial_prediction']) - 1
#     initial_prediction_list = []
#     cccc = 0
#     for iii in range(len(data_list)):
#         cccc += 1
#         if cccc == max_cccc or max_cccc == 0:
#             cccc = 0
#         import copy
#         answer = copy.deepcopy(data_list[iii]['answer'])
#         initial_prediction_of_another_question = initial_prediction_list_total['initial_prediction'][cccc]
#         answer= initial_prediction_of_another_question.replace('INFERENCE HERE', '')

#         initial_prediction_list.append(answer)



#     token_len_list = []
#     perplexity_list = []
#     answer_char_count_total = 0
#     for i, item in enumerate(data_list):
#         question = item['question']
#         original_question = item['original_question']
#         answer = item['answer']
#         initial_prediction_of_another_question = initial_prediction_list[i]

#         question = \
# f"""Question: {question}

# We have an inference example below to show you how to solve the problem with the proper style. please follow the style and solve the problem


# inference example: {initial_prediction_of_another_question}


# The style you have follow including the noticable language pattern, inference style, language style. In other words, make your solution as similar to the inference example as possible.

# If the inference process does not follow at the prediction before, you have to correct your style at anytime when you notice the style is not following the inference example. this is the most important requirement. please follow it.

# now, according to the inference example, please solve the problem. 
# """
        

        
#         answer_char_count_total += len(answer)

#         # Formatting the question based on the model type
#         if 'mistral' in model_name:
#             formatted_question = f"[INST] {question} [/INST]"
#         else:
#             formatted_question = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
        

#         q_and_a = formatted_question + ' ' + answer
#         answer_inputs = tokenizer(answer, return_tensors="pt")



#         answer_input_ids = answer_inputs['input_ids']
#         answer_input_ids = answer_input_ids.to(device)

#         # Decoding each token ID into its corresponding token
#         tokens = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in answer_input_ids[0]]
#         token_len = len(tokens)
#         token_len_list.append(token_len)
#         # print(tokens)

#         q_and_a_inputs = tokenizer(q_and_a, return_tensors="pt")
#         q_a_input_ids = q_and_a_inputs["input_ids"].to(device)
#         question_inputs = tokenizer(formatted_question, return_tensors="pt")
#         question_len = question_inputs["input_ids"].shape[1]
#         labels = q_a_input_ids.clone()
#         labels[:, :question_len] = -100

#         with torch.no_grad():
#             outputs = model(input_ids=q_a_input_ids, labels=labels)
#             loss = outputs.loss
#             perplexity = torch.exp(loss)
#             perplexity_list.append(perplexity.item())
            
#     average_perplexity = sum(perplexity_list) / len(perplexity_list) if perplexity_list else float('inf')
#     return average_perplexity



import torch, math
from copy import deepcopy

def get_max_len(cfg, default=4096):
    return getattr(cfg, "n_positions",
           getattr(cfg, "max_position_embeddings", default))

def in_context_perplexity_calculation_length_cutoff(
        data_list,
        model, tokenizer, model_name,
        chunk_size,                       # 50 or any value
        initial_prediction_list_total,
        device="cuda"):

    # ---------- build initial_prediction_list ----------
    max_cccc = len(initial_prediction_list_total["initial_prediction"]) - 1
    init_pred_list, cccc = [], 0
    for _ in range(len(data_list)):
        cccc = 0 if (cccc == max_cccc or max_cccc == 0) else cccc + 1
        init_pred = initial_prediction_list_total["initial_prediction"][cccc]
        init_pred_list.append(init_pred.replace("INFERENCE HERE", ""))

    max_len_main = get_max_len(model.config)
    ppl_list = []

    # tpl_first = """Question: {question}"""

    # ---------- prompt templates ----------
    tpl_first = """Question: {question}

# We have an inference example below to show you how to solve the problem with the proper style. please follow the style and solve the problem

# inference example: {style_ex}

# The style you have follow including the noticable language pattern, inference style, language style. In other words, make your solution as similar to the inference example as possible.

# If the inference process does not follow at the prediction before, you have to correct your style at anytime when you notice the style is not following the inference example. this is the most important requirement. please follow it.

# now, according to the inference example, please solve the problem.
# """

    tpl_follow = tpl_first + """

previous process {previous_process}
here is the previous inference process. please solve the problem after this process with the similar style in the inference example
"""

    for i, item in enumerate(data_list):
        question = item["question"]
        answer   = item["answer"]
        style_ex = init_pred_list[i]

        # ---- encode answer & prepend ONE space to match original concat ----
        ans_ids = tokenizer(" " + answer, return_tensors="pt").input_ids.to(device)
        ans_len = ans_ids.size(1)

        prev_text, sum_nll, tot_tok = "", 0.0, 0

        for s in range(0, ans_len, chunk_size):
            e = min(ans_len, s + chunk_size)
            chunk_ids = ans_ids[:, s:e]                        # [1, ≤chunk_size]
            chunk_text = tokenizer.decode(
                chunk_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            if prev_text == "":
                prompt = tpl_first.format(question=question, style_ex=style_ex)
            else:
                prompt = tpl_follow.format(
                    question=question, style_ex=style_ex, previous_process=prev_text.strip()
                )

            # wrap prompt
            if "mistral" in model_name.lower():
                formatted_q = f"[INST] {prompt} [/INST]"
            else:
                formatted_q = (
                    "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
                    f"{prompt} [/INST]"
                )

            enc_q = tokenizer(
                formatted_q,
                return_tensors="pt",
                truncation=True,
                max_length=max_len_main
            ).to(device)
            q_ids = enc_q.input_ids
            q_len = q_ids.size(1)

            chunk_input_ids = torch.cat([q_ids, chunk_ids], dim=1)
            labels = chunk_input_ids.clone()
            labels[:, :q_len] = -100

            with torch.no_grad():
                loss = model(input_ids=chunk_input_ids, labels=labels).loss.item()
                sum_nll += loss * chunk_ids.size(1)

            tot_tok += chunk_ids.size(1)
            prev_text += " " + chunk_text

        ppl_list.append(math.exp(sum_nll / tot_tok))

    return sum(ppl_list) / len(ppl_list) if ppl_list else float("inf")




def contrastive_perplexity_calculation(data_list, model, tokenizer, model_name, weak_model, weak_model_tokenizer, device='cuda'):
    perplexity_list = []
    answer_char_count_total = 0
    for problem_id, item in enumerate(data_list):
        question = item['question']
        original_question = item['original_question']
        answer = item['answer']
        answer_char_count_total += len(answer)

        # Formatting the question based on the model type
        if 'mistral' in model_name:
            formatted_question = f"[INST] {question} [/INST]"
            original_question = f"[INST] {original_question} [/INST]"
        else:
            formatted_question = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
        q_and_a = formatted_question + ' ' + answer

        q_and_a_inputs = tokenizer(q_and_a, return_tensors="pt")
        q_a_input_ids = q_and_a_inputs["input_ids"].to(device)
        question_inputs = tokenizer(formatted_question, return_tensors="pt")
        question_len = question_inputs["input_ids"].shape[1]
        labels = q_a_input_ids.clone()
        labels[:, :question_len] = -100

        gpt2_q_and_a_inputs = weak_model_tokenizer(q_and_a, return_tensors="pt")
        gpt2_q_a_input_ids = gpt2_q_and_a_inputs["input_ids"].to(device)
        gpt2_question_inputs = weak_model_tokenizer(formatted_question, return_tensors="pt")
        gpt2_question_len = gpt2_question_inputs["input_ids"].shape[1]
        gpt2_labels = gpt2_q_a_input_ids.clone()
        gpt2_labels[:, :gpt2_question_len] = -100

        with torch.no_grad():
            outputs = model(input_ids=q_a_input_ids, labels=labels)
            loss = outputs.loss
            perplexity = torch.exp(loss)

            gpt2_outputs = weak_model(input_ids=gpt2_q_a_input_ids, labels=gpt2_labels)
            gpt2_loss = gpt2_outputs.loss
            gpt2_perplexity = torch.exp(gpt2_loss)

            contrastive_perplexity = perplexity.item()/gpt2_perplexity.item()
            perplexity_list.append(contrastive_perplexity)
            
    average_perplexity = sum(perplexity_list) / len(perplexity_list) if perplexity_list else float('inf')
    return average_perplexity


def weak_model_perplexity_calculation(data_list, model, tokenizer, model_name, device='cuda'):
    perplexity_list = []
    answer_char_count_total = 0
    for problem_id, item in enumerate(data_list):
        question = item['question']
        original_question = item['original_question']
        answer = item['answer']
        answer_char_count_total += len(answer)

        # Formatting the question based on the model type
        if 'mistral' in model_name:
            formatted_question = f"[INST] {question} [/INST]"
            original_question = f"[INST] {original_question} [/INST]"
        else:
            formatted_question = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
        q_and_a = formatted_question + ' ' + answer

        q_and_a_inputs = tokenizer(q_and_a, return_tensors="pt")
        q_a_input_ids = q_and_a_inputs["input_ids"].to(device)
        question_inputs = tokenizer(formatted_question, return_tensors="pt")
        question_len = question_inputs["input_ids"].shape[1]
        labels = q_a_input_ids.clone()
        labels[:, :question_len] = -100
        # weak_model_tokenizer.encode(weak_model_tokenizer.eos_token)[0]

        with torch.no_grad():
            outputs = model(input_ids=q_a_input_ids, labels=labels)
            loss = outputs.loss
            perplexity = torch.exp(loss)
            perplexity_list.append(perplexity.item())
            
    average_perplexity = sum(perplexity_list) / len(perplexity_list) if perplexity_list else float('inf')
    return average_perplexity


def probability_calculation(data_list, model, tokenizer, model_name, device='cuda'):
    token_len_list = []

    token_prob_list = []
    answer_char_count_total = 0
    for problem_id, item in enumerate(data_list):
        question = item['question']
        answer = item['answer']
        answer_char_count_total += len(answer)
        pading_id = -100

        # Formatting the question based on the model type
        if 'mistral' in model_name:
            formatted_question = f"[INST] {question} [/INST]"
        elif 'gpt2' in model_name:
            formatted_question = f"{question}"
        else:
            formatted_question = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
        
        q_and_a = formatted_question + ' ' + answer
        answer_inputs = tokenizer(answer, return_tensors="pt")
        answer_input_ids = answer_inputs['input_ids']
        answer_input_ids = answer_input_ids.to(device)            

        # Decoding each token ID into its corresponding token
        tokens = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in answer_input_ids[0]]
        token_len = len(tokens)
        token_len_list.append(token_len)
        # print(tokens)

        q_and_a_inputs = tokenizer(q_and_a, return_tensors="pt")
        q_a_input_ids = q_and_a_inputs["input_ids"].to(device)
        question_inputs = tokenizer(formatted_question, return_tensors="pt")
        question_len = question_inputs["input_ids"].shape[1]
        labels = q_a_input_ids.clone()
        labels[:, :question_len] = pading_id

        token_prob_list_item = []
        with torch.no_grad():
            outputs = model(input_ids=q_a_input_ids, labels=labels)
            logits  = outputs.logits                     # [B, L, V]

            # 1️⃣  对齐：丢掉最后 1 个 logit，丢掉第 1 个 label
            logits_shift = logits[:, :-1, :]             # 预测位置 t+1
            labels_shift = labels[:, 1:]                 # 真值位置 t+1

            # 2️⃣  softmax → 概率
            probs = F.softmax(logits_shift, dim=-1)      # [B, L-1, V]

            # 3️⃣  mask padding，并取 ground‑truth 概率
            valid_mask = labels_shift != pading_id            # [B, L-1]
            labels_g = labels_shift.clone()
            labels_g[~valid_mask] = 0                    # 临时占位

            label_probs = probs.gather(2, labels_g.unsqueeze(-1)) \
                            .squeeze(-1)             # [B, L-1]
            label_probs = label_probs[valid_mask]       # 1‑D 有效概率

            # 4️⃣  打印每个 label 和其对应的概率
            probs_list = []
            for i, (label, prob) in enumerate(zip(labels_shift[valid_mask], label_probs)):
                token_prob_list_item.append(prob)
                probs_list.append(prob)
        token_prob_list.append(probs_list)
    return token_prob_list


import torch
import torch.nn.functional as F

# ========== 3. 概率计算 ==========
def probability_calculation_gpt2(
        data_list, 
        model, 
        tokenizer, 
        model_name: str, 
        device: str = "cuda",
        ctx_len: int = 1024,           # GPT-2 最大上下文
        stride:  int = 512,            # 每次向右滑动步长
    ):
    """
    计算 answer 区段的逐 token 概率；适配超长输入。
    返回 [[p1, p2, ...], ...]，与 data_list 顺序一致。
    """
    pad_id = -100
    results = []

    for item in data_list:
        question = item["question"]
        answer   = item["answer"]

        # === 拼 prompt（保留你原来的格式判断） ===
        lname = model_name.lower()
        if "mistral" in lname:
            prompt = f"[INST] {question} [/INST]"
        elif "gpt2" in lname:
            prompt = question
        else:
            prompt = ("[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
                      f"{question} [/INST]")

        # === 直接拿整句 token id，不让 tokenizer 截断 ===
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full_ids   = prompt_ids + tokenizer.encode(" " + answer, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        answer_probs = []                       # 结果累计

        # === 滑动窗口 ===
        for start in range(0, len(full_ids), stride):
            end = min(start + ctx_len, len(full_ids))
            ids_slice = full_ids[start:end]

            # input_ids
            input_ids = torch.tensor([ids_slice], device=device)

            # labels：默认监督整段，再按需要屏蔽
            labels = input_ids.clone()

            # ① 屏蔽问句（全局坐标 < prompt_len）
            global_pos = torch.arange(start, end, device=device)
            labels[0][global_pos < prompt_len] = pad_id

            # ② 只让“新进入窗口”的 token 计算 loss
            if end != len(full_ids):
                labels[0][: -stride] = pad_id

            # 前向
            with torch.no_grad():
                logits = model(input_ids, labels=labels).logits

            logits_shift = logits[:, :-1, :]
            labels_shift = labels[:, 1:]
            mask_valid   = labels_shift != pad_id

            probs = F.softmax(logits_shift, dim=-1)
            tok_probs = probs.gather(
                2, labels_shift.masked_fill(~mask_valid, 0).unsqueeze(-1)
            ).squeeze(-1)[mask_valid].tolist()

            answer_probs.extend(tok_probs)

            if end == len(full_ids):
                break

        # 最后仅保留 answer 部分概率
        answer_token_num = len(full_ids) - prompt_len
        results.append(answer_probs[-answer_token_num:])

    return results


import torch
import torch.nn.functional as F

def topk_entropy_calculation(
        data_list,
        model,
        tokenizer,
        model_name,
        k=10,                          # 前 k 个词
        device='cuda',
        return_token_level=False):     # 是否返回每个 token 的熵
    padding_id = -100
    sample_entropy_list = []          # 存每条样本的平均熵
    token_entropy_list  = []          # 可选：存每个 token 的熵

    for item in data_list:
        question = item['question']
        answer   = item['answer']

        # ---------- 与原函数同样的格式化 ----------
        if 'mistral' in model_name:
            formatted_q = f"[INST] {question} [/INST]"
        elif 'gpt2' in model_name:
            formatted_q = question
        else:
            formatted_q = (
                "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
                f"{question} [/INST]"
            )
        qa_text = formatted_q + ' ' + answer

        qa_inputs     = tokenizer(qa_text, return_tensors='pt').to(device)
        question_len  = tokenizer(formatted_q, return_tensors='pt')['input_ids'].shape[1]

        labels = qa_inputs['input_ids'].clone()
        labels[:, :question_len] = padding_id

        with torch.no_grad():
            logits = model(**qa_inputs).logits          # [B, L, V]
        logits_shift  = logits[:, :-1, :]               # 预测位置 t+1
        labels_shift  = labels[:, 1:]

        valid_mask = labels_shift != padding_id         # 取 answer 部分

        # ---------- 计算 top-k 概率 ----------
        probs = F.softmax(logits_shift, dim=-1)         # [B, L-1, V]
        topk_probs, _ = probs.topk(k, dim=-1)           # [B, L-1, k]
        topk_probs = topk_probs / topk_probs.sum(-1, keepdim=True)  # 重新归一化

        # ---------- 熵 ----------
        token_entropy = -(topk_probs * topk_probs.log()).sum(-1)    # [B, L-1]
        token_entropy = token_entropy[valid_mask]        # 只保留有效 token

        # ---------- 聚合 ----------
        sample_entropy_list.append(token_entropy.mean().item())
        if return_token_level:
            token_entropy_list.append(token_entropy.cpu().tolist())

    if return_token_level:
        return sample_entropy_list, token_entropy_list
    return sample_entropy_list




def calibrated_perplexity_with_threshold_calculation(token_prob_list, token_entropy_list, threshold):
    perplexity_list = []

    for label_probs, label_entropy in zip(token_prob_list, token_entropy_list):
        filtered_probs = [p for p, ent in zip(label_probs, label_entropy) if ent >= threshold]

        label_probs = torch.tensor(filtered_probs)      # ≈ -0.92
        avg_log_p = label_probs.log().mean()
        ppl = (-avg_log_p).exp()                    # ≈ 2.5

        # print("avg p:", avg_log_p.exp().item(),
        #     "PPL:",  ppl.item())
        perplexity_list.append(ppl)

    average_perplexity = sum(perplexity_list) / len(perplexity_list) if perplexity_list else float('inf')
    return average_perplexity.item()



def calibrated_perplexity_calculation_given_probability(token_prob_list, function_template = None, reference_token_prob_list = None):
    perplexity_list = []

    if not reference_token_prob_list:
        length_token_list = token_prob_list
    else:
        length_token_list = reference_token_prob_list
    for problem_id, label_probs in enumerate(length_token_list):
        # 4️⃣  打印每个 label 和其对应的概率
        total_length = 0
        total_length_list = []
        for prob in label_probs:
            if function_template:
                if function_template == '1':
                    def f(x):
                        return np.exp(-1.0 / (-np.log(x)))
                elif function_template == '2':
                    def f(x):
                        if x < 0.95:
                            return 1-x
                        if x > 0.95 or x == 0.95:
                            return 0
                elif function_template == '3':
                    def f(x):
                        if x < 0.95:
                            return 1
                        if x > 0.95 or x == 0.95:
                            return 0
                elif function_template == '4':
                    def f(x):
                        if x < 0.99:
                            return 1
                        if x > 0.99 or x == 0.99:
                            return 0
                elif function_template == '5':
                    def f(x):
                        if x < 0.1:
                            return 0
                        if x > 0.1 or x == 0.1:
                            return 1
                elif function_template == '6':
                    def f(x):
                        if x < 0.1:
                            return 1
                        if x > 0.1 or x == 0.1 and x < 0.98:
                            return 1
                        if x > 0.98 or x == 0.98:
                            return 0
                elif function_template == 'no_calibration':
                    def f(x):
                        return 1
                try:
                    prob = prob.to('cpu')
                except:
                    a = 1

                if function_template == '7':
                    def f(x):
                        if x > 0.98 or x == 0.98:
                            return 0
                        else:
                            return 1
                converted_length = f(prob)
                # if converted_length > 1:
                #     converted_length = 0
                total_length += converted_length
                # N_list.append(f(prob))
                total_length_list.append(converted_length)
                # calibrated_probs_list.append(prob)

        # 5️⃣  验证平均概率和 PPL 是否对得上
        token_prob_list_item = token_prob_list[problem_id]
        label_probs = torch.tensor(token_prob_list_item)      # ≈ -0.92
        if not function_template:
            avg_log_p = label_probs.log().mean()        # ≈ -0.92
        else:
            avg_log_p = sum(label_probs.log())/total_length        # ≈ -0.92
        ppl = (-avg_log_p).exp()                    # ≈ 2.5

        if ppl > 100:
            a = 1
        # print("avg p:", avg_log_p.exp().item(),
        #     "PPL:",  ppl.item())
        perplexity_list.append(ppl)

    average_perplexity = sum(perplexity_list) / len(perplexity_list) if perplexity_list else float('inf')
    return average_perplexity.item()


def calibrated_perplexity_calculation(data_list, model, tokenizer, model_name, device='cuda', function_template = None):
    token_len_list = []
    perplexity_list = []
    answer_char_count_total = 0
    for problem_id, item in enumerate(data_list):
        question = item['question']
        original_question = item['original_question']
        answer = item['answer']
        answer_char_count_total += len(answer)

        # Formatting the question based on the model type
        if 'mistral' in model_name:
            formatted_question = f"[INST] {question} [/INST]"
            original_question = f"[INST] {original_question} [/INST]"
        else:
            formatted_question = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
        
        q_and_a = formatted_question + ' ' + answer
        answer_inputs = tokenizer(answer, return_tensors="pt")
        answer_input_ids = answer_inputs['input_ids']
        answer_input_ids = answer_input_ids.to(device)            

        # Decoding each token ID into its corresponding token
        tokens = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in answer_input_ids[0]]
        token_len = len(tokens)
        token_len_list.append(token_len)
        # print(tokens)

        q_and_a_inputs = tokenizer(q_and_a, return_tensors="pt")
        q_a_input_ids = q_and_a_inputs["input_ids"].to(device)
        question_inputs = tokenizer(formatted_question, return_tensors="pt")
        question_len = question_inputs["input_ids"].shape[1]
        labels = q_a_input_ids.clone()
        labels[:, :question_len] = -100

        with torch.no_grad():
            outputs = model(input_ids=q_a_input_ids, labels=labels)
            logits  = outputs.logits                     # [B, L, V]

            # 1️⃣  对齐：丢掉最后 1 个 logit，丢掉第 1 个 label
            logits_shift = logits[:, :-1, :]             # 预测位置 t+1
            labels_shift = labels[:, 1:]                 # 真值位置 t+1

            # 2️⃣  softmax → 概率
            probs = F.softmax(logits_shift, dim=-1)      # [B, L-1, V]

            # 3️⃣  mask padding，并取 ground‑truth 概率
            valid_mask = labels_shift != -100            # [B, L-1]
            labels_g = labels_shift.clone()
            labels_g[~valid_mask] = 0                    # 临时占位

            label_probs = probs.gather(2, labels_g.unsqueeze(-1)) \
                            .squeeze(-1)             # [B, L-1]
            label_probs = label_probs[valid_mask]       # 1‑D 有效概率

            # 4️⃣  打印每个 label 和其对应的概率
            calibrated_probs_list = []
            total_length = 0
            total_length_list = []
            for i, (label, prob) in enumerate(zip(labels_shift[valid_mask], label_probs)):
                two_decimals = round(prob.item(), 2)  
                token = tokenizer.decode(label.item())   # 将 id 转为 token
                # print(f"Label: {token}, Probability: {prob.item():.4e}")
                
                if not function_template:
                    if two_decimals < 1:
                        # print(f"Label: {token}           {prob.item():.4e}")
                        calibrated_probs_list.append(prob)
                else:
                    if function_template == '1':
                        def f(x):
                            return np.exp(-1.0 / (-np.log(x)))
                    elif function_template == '2':
                        def f(x):
                            if x < 0.95:
                                return 1-x
                            if x > 0.95 or x == 0.95:
                                return 0
                    elif function_template == '3':
                        def f(x):
                            if x < 0.95:
                                return 1
                            if x > 0.95 or x == 0.95:
                                return 0
                    elif function_template == '4':
                        def f(x):
                            if x < 0.99:
                                return 1
                            if x > 0.99 or x == 0.99:
                                return 0
                    elif function_template == '5':
                        def f(x):
                            if x < 0.1:
                                return 0
                            if x > 0.1 or x == 0.1:
                                return 1
                    elif function_template == '6':
                        def f(x):
                            if x < 0.1:
                                return 0
                            if x > 0.1 or x == 0.1 and x < 0.98:
                                return 1
                            if x > 0.98 or x == 0.98:
                                return 0
                    elif function_template == 'no_calibration':
                        def f(x):
                            return 1
                    prob = prob.to('cpu')
                    converted_length = f(prob)
                    if converted_length > 1:
                        converted_length = 0
                    total_length += converted_length
                    # N_list.append(f(prob))
                    total_length_list.append(converted_length)
                    calibrated_probs_list.append(prob)



                # else:
                #     print(f"Label: {token}           1.00")

            # 5️⃣  验证平均概率和 PPL 是否对得上
            label_probs = torch.tensor(calibrated_probs_list)      # ≈ -0.92
            if not function_template:
                avg_log_p = label_probs.log().mean()        # ≈ -0.92
            else:
                avg_log_p = sum(label_probs.log())/total_length        # ≈ -0.92
            ppl = (-avg_log_p).exp()                    # ≈ 2.5
            # print("avg p:", avg_log_p.exp().item(),
            #     "PPL:",  ppl.item())
            perplexity_list.append(ppl)

    average_perplexity = sum(perplexity_list) / len(perplexity_list) if perplexity_list else float('inf')
    return average_perplexity.item()

import torch
import math

def probability_in_context_perplexity_calculation(
        data_list,
        model,
        tokenizer,
        model_name,
        device="cuda"):
    """
    返回
    -------
    avg_ppl : float
        传统的平均 Perplexity。
    avg_seq_prob : float
        整段 answer 的联合概率（已做下溢保护）。
    avg_log_seq_prob : float
        整段 answer 的对数联合概率（推荐使用，绝不会下溢）。
    """
    ppl_list          = []
    seq_prob_list     = []
    log_seq_prob_list = []

    for item in data_list:
        question = item["question"]
        answer   = item["answer"]


        # ---------- 组装 prompt ----------
        if "mistral" in model_name:
            formatted_question = f"[INST] {question} [/INST]"
        else:
            formatted_question = (
                "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
                f"{question} [/INST]"
            )
        prompt_and_answer = formatted_question + " " + answer

        # ---------- 编码 ----------
        qa_inputs = tokenizer(prompt_and_answer, return_tensors="pt").to(device)
        labels    = qa_inputs.input_ids.clone()

        question_len = tokenizer(formatted_question,
                                 return_tensors="pt").input_ids.shape[1]
        labels[:, :question_len] = -100      # 只在 answer 部分计算损失

        # ---------- 前向 ----------
        with torch.no_grad():
            out   = model(input_ids=qa_inputs.input_ids, labels=labels)
            loss  = out.loss                 # mean NLL (nats, float32)
            N     = (labels != -100).sum()   # answer token 数 (tensor)

            # ----- Perplexity -----
            ppl = torch.exp(loss)            # e^{loss}

            # ----- 联合概率 -----
            # log_prob = -loss * N
            log_seq_prob = (-loss.item()) * N.item()   # python float64
            # 防止下溢: float64 的 exp(-745) ≈ 5e-324，再小就变 0
            seq_prob = math.exp(log_seq_prob) if log_seq_prob > -745 else 0.0

            # ----- 收集 -----
            ppl_list.append(ppl.item())
            seq_prob_list.append(seq_prob)
            log_seq_prob_list.append(log_seq_prob)

    return ppl_list, seq_prob_list, log_seq_prob_list

    # avg_ppl          = sum(ppl_list)          / len(ppl_list)
    # avg_seq_prob     = sum(seq_prob_list)     / len(seq_prob_list)
    # avg_log_seq_prob = sum(log_seq_prob_list) / len(log_seq_prob_list)
    
    # return avg_ppl, avg_log_seq_prob

def original_perplexity_calculation(data_list, model, tokenizer, model_name, device='cuda'):
    token_len_list = []
    perplexity_list = []
    IDF_list = []
    loss_list = []
    answer_char_count_total = 0
    for problem_id, item in enumerate(data_list):
        question = item['question']
        original_question = item['original_question']
        answer = item['answer']
        answer_char_count_total += len(answer)

        # Formatting the question based on the model type
        if 'mistral' in model_name:
            formatted_question = f"[INST] {original_question} [/INST]"
        else:
            formatted_question = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{original_question} [/INST]"
        
        q_and_a = formatted_question + ' ' + answer
        answer_inputs = tokenizer(answer, return_tensors="pt")
        answer_input_ids = answer_inputs['input_ids']
        answer_input_ids = answer_input_ids.to(device)

        # Decoding each token ID into its corresponding token
        tokens = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in answer_input_ids[0]]
        token_len = len(tokens)
        token_len_list.append(token_len)
        # print(tokens)

        q_and_a_inputs = tokenizer(q_and_a, return_tensors="pt")
        q_a_input_ids = q_and_a_inputs["input_ids"].to(device)
        question_inputs = tokenizer(formatted_question, return_tensors="pt")
        question_len = question_inputs["input_ids"].shape[1]
        labels = q_a_input_ids.clone()
        labels[:, :question_len] = -100

        with torch.no_grad():
            outputs = model(input_ids=q_a_input_ids, labels=labels)
            loss = outputs.loss
            perplexity = torch.exp(loss)
            loss_list.append(loss.item())

            # Forward pass: answer only (to get perplexity of answer alone)
            # Here, we compute loss on answer_input_ids starting from the second token
            
            if len(answer_input_ids[0]) > 1:
                outputs_y_only = model(input_ids=answer_input_ids, labels=answer_input_ids)
                y_only_loss = outputs_y_only.loss

                # Compute IDF
                IDF = loss / y_only_loss
                IDF_list.append(IDF.item())

            perplexity_list.append(perplexity.item())
            
            if perplexity_list == []:
                perplexity_list.append(1000000)

    return perplexity_list, IDF_list, loss_list, token_len_list
    # average_perplexity = sum(perplexity_list) / len(perplexity_list) if perplexity_list else float('inf')
    # average_IDF = sum(IDF_list) / len(IDF_list) if IDF_list else float('inf')
    # average_loss = sum(loss_list) / len(loss_list) if loss_list else float('inf')
    # average_token_len = sum(token_len_list) / len(token_len_list)
    # average_char_len = answer_char_count_total / len(data_list)

    # return average_perplexity, average_IDF, average_loss, average_token_len, average_char_len




def customized_perplexity_calculation(data_list, model, tokenizer, model_name, device='cuda'):
    token_len_list = []
    perplexity_list = []
    perplexity_list_original = []
    loss_list = []
    answer_char_count_total = 0
    for problem_id, item in enumerate(data_list):
        question = item['question']
        original_question = item['original_question']
        answer = item['answer']
        answer_char_count_total += len(answer)
        # Formatting the question based on the model type
        if 'mistral' in model_name:
            formatted_question = f"[INST] {question} [/INST]"
            original_question = f"[INST] {original_question} [/INST]"
        else:
            formatted_question = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
        
        q_and_a = formatted_question + ' ' + answer
        answer_inputs = tokenizer(answer, return_tensors="pt")
        answer_input_ids = answer_inputs['input_ids']
        answer_input_ids = answer_input_ids.to(device)

        # Decoding each token ID into its corresponding token
        tokens = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in answer_input_ids[0]]
        token_len = len(tokens)
        token_len_list.append(token_len)
        # print(tokens)

        q_and_a_inputs = tokenizer(q_and_a, return_tensors="pt")
        q_a_input_ids = q_and_a_inputs["input_ids"].to(device)
        question_inputs = tokenizer(formatted_question, return_tensors="pt")
        question_len = question_inputs["input_ids"].shape[1]
        labels = q_a_input_ids.clone()
        labels[:, :question_len] = -100

        

        with torch.no_grad():
            # Function to get token embedding from the model
            def get_token_embedding(token_id, model):
                # Ensure the token_id is valid
                vocab_size = model.config.vocab_size
                if token_id >= vocab_size or token_id < 0:
                    raise ValueError(f"Token ID {token_id} is out of bounds. Vocab size is {vocab_size}.")
                
                # Convert token_id to input_ids
                input_ids = torch.tensor([[token_id]])  # Shape: (1, 1) for a single token
                input_ids = input_ids.to(device)  # Ensure the input tensor is on the correct device
                
                # Pass the input through the model to get the hidden states
                outputs = model(input_ids=input_ids, output_hidden_states=True)  # Enable hidden states
                hidden_states = outputs.hidden_states  # hidden_states is a tuple with all hidden layers' states
                
                # Get the embedding for the token from the last hidden state (using the last layer's output)
                token_embedding = hidden_states[-1][0, 0]  # Shape: (hidden_size,)
                
                return token_embedding

            # Forward pass
            outputs = model(input_ids=q_a_input_ids, labels=labels)
            logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)


            # Get probabilities by applying softmax
            probabilities = torch.softmax(logits, dim=-1)  # shape: (batch_size, seq_len, vocab_size)



            loss = outputs.loss
            loss_list.append(loss.item())
            perplexity = torch.exp(loss)
            perplexity_list_original.append(perplexity)


            # If you're predicting tokens (in inference mode), you may want to get the predicted tokens
            predicted_tokens = torch.argmax(probabilities, dim=-1)  # shape: (batch_size, seq_len)

            # Calculate semantic similarity and sum probabilities
            for batch_idx in range(probabilities.size(0)):  # Loop over batch size
                cross_entropy = 0
                response_counter = 0
                for seq_idx in range(probabilities.size(1)):  # Loop over sequence length
                    # Get the top 5 predicted token indices and their corresponding probabilities
                    top5_probs, top5_indices = torch.topk(probabilities[batch_idx, seq_idx], 5)  # shape: (5,)
                    # Get the ground truth token
                    if seq_idx + 1 < probabilities.size(1):
                        ground_truth_token = labels[batch_idx, seq_idx + 1]  # Get the ground truth token ID
                        if ground_truth_token != -100:
                            
                            match_prob = probabilities[batch_idx, seq_idx, ground_truth_token].item()  # Add the true ground truth probability

                            if match_prob > 0.01:
                                response_counter += 1
                                # print('groundtruth_probability: ', match_prob)
                                log_prob = torch.log(torch.tensor(match_prob, dtype=torch.float32, device=device))
                                cross_entropy -= log_prob
                            # else:
                            #     response_counter += 1
                            #     # log_prob = torch.log(torch.tensor(match_prob, dtype=torch.float32, device=device))
                            #     log_prob = torch.log(torch.tensor(0.01, dtype=torch.float32, device=device))
                            #     cross_entropy -= log_prob
                            #     # print('groundtruth_probability: ', match_prob)



                        #     
                        #     ground_truth_token_string = tokenizer.decode([ground_truth_token.item()])  # Decoding ground truth token ID to string
                            
                        #     # Get the embedding of the ground truth token
                        #     ground_truth_embedding = get_token_embedding(ground_truth_token, model)

                        #     # Print the ground truth token
                        #     print(f"Batch {batch_idx}, Token {seq_idx}: Ground Truth Token ID {ground_truth_token.item()} -> '{ground_truth_token_string}'")

                        #     # Initialize a variable to accumulate the sum of probabilities for matching tokens
                        #     match_prob = probabilities[batch_idx, seq_idx, ground_truth_token].item()  # Add the true ground truth probability
                        #     print('groundtruth_probability: ', match_prob)
                        #     # Print the top 5 tokens and their probabilities
                        #     print(f"  Top 5 Predictions:")
                        #     for i in range(5):
                        #         predicted_token_id = top5_indices[i]
                        #         predicted_token_prob = top5_probs[i]

                        #         if ground_truth_token == predicted_token_id:
                        #             predicted_token_string = tokenizer.decode([predicted_token_id.item()]) 
                        #             print(f"  {i+1}. Token ID {predicted_token_id.item()} -> '{predicted_token_string}' with Probability {predicted_token_prob.item()}")
                        #             continue
                        #         else:

                        #             # Convert the predicted token ID to the corresponding token string
                        #             predicted_token_string = tokenizer.decode([predicted_token_id.item()])  # Decoding token ID to string

                        #             # Get the embedding of the predicted token
                        #             predicted_token_embedding = get_token_embedding(predicted_token_id, model)

                        #             # Calculate cosine similarity between the ground truth and predicted token
                        #             similarity = cosine_similarity(ground_truth_embedding.unsqueeze(0).cpu().detach().numpy(),
                        #                                         predicted_token_embedding.unsqueeze(0).cpu().detach().numpy())[0][0]

                        #             print(f"  {i+1}. Token ID {predicted_token_id.item()} -> '{predicted_token_string}' with Probability {predicted_token_prob.item()} and Similarity {similarity:.4f}")

                        #             # If similarity is high, add the probability to match_prob
                        #             if similarity > 0.4:  # You can adjust this threshold as needed
                        #                 match_prob += predicted_token_prob.item()

                        #     log_prob = torch.log(torch.tensor(match_prob, dtype=torch.float32, device=device))
                        #     cross_entropy -= log_prob
                            
                        #     # print(f"  Total Match Probability for Ground Truth '{ground_truth_token_string}': {match_prob:.4f}")
                        #     # print(f"  Total Match Probability for Ground Truth '{ground_truth_token_string}': {match_prob}")
                        #     print(f"  Total Match Probability for Ground Truth '{ground_truth_token_string}': {log_prob}")
                        #     a = 1
                        # else:
                        #     print(f"Batch {batch_idx}, Token {seq_idx}: Ground Truth Token ID {ground_truth_token.item()} (Ignored as padding)")

                average_cross_entropy = cross_entropy/response_counter
                perplexity = torch.exp(average_cross_entropy) 
                perplexity_list.append(perplexity)
            a = 1

    average_perplexity = sum(perplexity_list) / len(perplexity_list) if perplexity_list else float('inf')
    average_perplexity_original = sum(perplexity_list_original) / len(perplexity_list_original) if perplexity_list_original else float('inf')
    average_token_len = sum(token_len_list) / len(token_len_list)
    average_loss = sum(loss_list) / len(loss_list) if loss_list else float('inf')
    average_char_len = answer_char_count_total / len(data_list)
    return average_perplexity, average_perplexity_original, average_token_len, average_char_len, average_loss, cross_entropy


def assumulated_perplexity_calculation(data_list, model, tokenizer, model_name, similarity_compare_to_irrelevant_prediction, device='cuda'):
    token_len_list = []
    perplexity_list = []
    perplexity_list_original = []
    loss_list = []
    answer_char_count_total = 0
    for problem_id, item in enumerate(data_list):
        question = item['question']
        original_question = item['original_question']
        answer = item['answer']
        answer_char_count_total += len(answer)
        if similarity_compare_to_irrelevant_prediction:
            a = 1

        # Formatting the question based on the model type
        if 'mistral' in model_name:
            formatted_question = f"[INST] {question} [/INST]"
            original_question = f"[INST] {original_question} [/INST]"
        else:
            formatted_question = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
        
        q_and_a = formatted_question + ' ' + answer
        answer_inputs = tokenizer(answer, return_tensors="pt")
        answer_input_ids = answer_inputs['input_ids']
        answer_input_ids = answer_input_ids.to(device)

        # Decoding each token ID into its corresponding token
        tokens = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in answer_input_ids[0]]
        token_len = len(tokens)
        token_len_list.append(token_len)
        # print(tokens)

        q_and_a_inputs = tokenizer(q_and_a, return_tensors="pt")
        q_a_input_ids = q_and_a_inputs["input_ids"].to(device)
        question_inputs = tokenizer(formatted_question, return_tensors="pt")
        question_len = question_inputs["input_ids"].shape[1]
        labels = q_a_input_ids.clone()
        labels[:, :question_len] = -100
        

        with torch.no_grad():
            # Function to get token embedding from the model
            def get_token_embedding(token_id, model):
                # Ensure the token_id is valid
                vocab_size = model.config.vocab_size
                if token_id >= vocab_size or token_id < 0:
                    raise ValueError(f"Token ID {token_id} is out of bounds. Vocab size is {vocab_size}.")
                
                # Convert token_id to input_ids
                input_ids = torch.tensor([[token_id]])  # Shape: (1, 1) for a single token
                input_ids = input_ids.to(device)  # Ensure the input tensor is on the correct device
                
                # Pass the input through the model to get the hidden states
                outputs = model(input_ids=input_ids, output_hidden_states=True)  # Enable hidden states
                hidden_states = outputs.hidden_states  # hidden_states is a tuple with all hidden layers' states
                
                # Get the embedding for the token from the last hidden state (using the last layer's output)
                token_embedding = hidden_states[-1][0, 0]  # Shape: (hidden_size,)
                
                return token_embedding

            # Forward pass
            outputs = model(input_ids=q_a_input_ids, labels=labels)
            logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)

            # Get probabilities by applying softmax
            probabilities = torch.softmax(logits, dim=-1)  # shape: (batch_size, seq_len, vocab_size)



            loss = outputs.loss
            loss_list.append(loss.item())
            perplexity = torch.exp(loss)
            perplexity_list_original.append(perplexity)


            # If you're predicting tokens (in inference mode), you may want to get the predicted tokens
            predicted_tokens = torch.argmax(probabilities, dim=-1)  # shape: (batch_size, seq_len)

            # Calculate semantic similarity and sum probabilities
            for batch_idx in range(probabilities.size(0)):  # Loop over batch size
                cross_entropy = 0
                response_counter = 0
                for seq_idx in range(probabilities.size(1)):  # Loop over sequence length
                    # Get the top 5 predicted token indices and their corresponding probabilities
                    top5_probs, top5_indices = torch.topk(probabilities[batch_idx, seq_idx], 5)  # shape: (5,)

                    # Get the ground truth token
                    if seq_idx + 1 < probabilities.size(1):
                        ground_truth_token = labels[batch_idx, seq_idx + 1]  # Get the ground truth token ID
                        if ground_truth_token != -100:
                            response_counter += 1
                            ground_truth_token_string = tokenizer.decode([ground_truth_token.item()])  # Decoding ground truth token ID to string
                            
                            # Get the embedding of the ground truth token
                            ground_truth_embedding = get_token_embedding(ground_truth_token, model)

                            # Print the ground truth token
                            print(f"Batch {batch_idx}, Token {seq_idx}: Ground Truth Token ID {ground_truth_token.item()} -> '{ground_truth_token_string}'")

                            # Initialize a variable to accumulate the sum of probabilities for matching tokens
                            match_prob = probabilities[batch_idx, seq_idx, ground_truth_token].item()  # Add the true ground truth probability
                            print('groundtruth_probability: ', match_prob)
                            # Print the top 5 tokens and their probabilities
                            print(f"  Top 5 Predictions:")
                            for i in range(5):
                                predicted_token_id = top5_indices[i]
                                predicted_token_prob = top5_probs[i]

                                if ground_truth_token == predicted_token_id:
                                    predicted_token_string = tokenizer.decode([predicted_token_id.item()]) 
                                    print(f"  {i+1}. Token ID {predicted_token_id.item()} -> '{predicted_token_string}' with Probability {predicted_token_prob.item()}")
                                    continue
                                else:

                                    # Convert the predicted token ID to the corresponding token string
                                    predicted_token_string = tokenizer.decode([predicted_token_id.item()])  # Decoding token ID to string

                                    # Get the embedding of the predicted token
                                    predicted_token_embedding = get_token_embedding(predicted_token_id, model)

                                    # Calculate cosine similarity between the ground truth and predicted token
                                    similarity = cosine_similarity(ground_truth_embedding.unsqueeze(0).cpu().detach().numpy(),
                                                                predicted_token_embedding.unsqueeze(0).cpu().detach().numpy())[0][0]

                                    print(f"  {i+1}. Token ID {predicted_token_id.item()} -> '{predicted_token_string}' with Probability {predicted_token_prob.item()} and Similarity {similarity:.4f}")

                                    # If similarity is high, add the probability to match_prob
                                    if similarity > 0.4:  # You can adjust this threshold as needed
                                        match_prob += predicted_token_prob.item()

                            log_prob = torch.log(torch.tensor(match_prob, dtype=torch.float32, device=device))
                            cross_entropy -= log_prob
                            
                            # print(f"  Total Match Probability for Ground Truth '{ground_truth_token_string}': {match_prob:.4f}")
                            # print(f"  Total Match Probability for Ground Truth '{ground_truth_token_string}': {match_prob}")
                            print(f"  Total Match Probability for Ground Truth '{ground_truth_token_string}': {log_prob}")
                            a = 1
                        else:
                            print(f"Batch {batch_idx}, Token {seq_idx}: Ground Truth Token ID {ground_truth_token.item()} (Ignored as padding)")

                average_cross_entropy = cross_entropy/response_counter
                perplexity = torch.exp(average_cross_entropy) 
                perplexity_list.append(perplexity)
            a = 1

    average_perplexity = sum(perplexity_list) / len(perplexity_list) if perplexity_list else float('inf')
    average_perplexity_original = sum(perplexity_list_original) / len(perplexity_list_original) if perplexity_list_original else float('inf')
    average_token_len = sum(token_len_list) / len(token_len_list)
    average_loss = sum(loss_list) / len(loss_list) if loss_list else float('inf')
    average_char_len = answer_char_count_total / len(data_list)
    return average_perplexity, average_perplexity_original, average_token_len, average_char_len, average_loss






def perplexity_calculation_redundant(data_list, model, tokenizer, model_name, device='cuda', not_cap_perplexity = True):
    token_len_list = []
    perplexity_list = []
    IDF_list = []
    loss_list = []
    answer_char_count_total = 0
    for problem_id, item in enumerate(data_list):
        original_question = item['question']
        answer = item['answer']
        answer_char_count_total += len(answer)

        # Formatting the question based on the model type
        if 'mistral' in model_name:
            formatted_question = f"[INST] {original_question} [/INST]"
        else:
            formatted_question = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{original_question} [/INST]"
        
        q_and_a = formatted_question + ' ' + answer
        answer_inputs = tokenizer(answer, return_tensors="pt")
        answer_input_ids = answer_inputs['input_ids']
        answer_input_ids = answer_input_ids.to(device)

        # Decoding each token ID into its corresponding token
        tokens = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in answer_input_ids[0]]
        token_len = len(tokens)
        token_len_list.append(token_len)
        # print(tokens)

        q_and_a_inputs = tokenizer(q_and_a, return_tensors="pt")
        q_a_input_ids = q_and_a_inputs["input_ids"].to(device)
        question_inputs = tokenizer(formatted_question, return_tensors="pt")
        question_len = question_inputs["input_ids"].shape[1]
        labels = q_a_input_ids.clone()
        labels[:, :question_len] = -100

        with torch.no_grad():
            outputs = model(input_ids=q_a_input_ids, labels=labels)
            loss = outputs.loss
            perplexity = torch.exp(loss)
            loss_list.append(loss.item())

            # Forward pass: answer only (to get perplexity of answer alone)
            # Here, we compute loss on answer_input_ids starting from the second token
            
            if len(answer_input_ids[0]) > 1:
                outputs_y_only = model(input_ids=answer_input_ids, labels=answer_input_ids)
                y_only_loss = outputs_y_only.loss
                y_only_perplexity = torch.exp(y_only_loss)

                # Compute IDF
                IDF = perplexity / y_only_perplexity
                IDF_list.append(IDF.item())

            if perplexity > 30:
                # a = 1
                if not_cap_perplexity:
                    perplexity_list.append(perplexity.item())
            else:
                perplexity_list.append(perplexity.item())
            
            if perplexity_list == []:
                perplexity_list.append(1000000)

    average_perplexity = sum(perplexity_list) / len(perplexity_list) if perplexity_list else float('inf')
    average_IDF = sum(IDF_list) / len(IDF_list) if IDF_list else float('inf')
    average_loss = sum(loss_list) / len(loss_list) if loss_list else float('inf')
    average_token_len = sum(token_len_list) / len(token_len_list)
    average_char_len = answer_char_count_total / len(data_list)

    return average_perplexity, average_IDF, average_loss, average_token_len, average_char_len



def in_context_perplexity_calculation_redundant(data_list, model, tokenizer, model_name, device='cuda', not_cap_perplexity = True):
    token_len_list = []
    perplexity_list = []
    IDF_list = []
    loss_list = []
    answer_char_count_total = 0
    for problem_id, item in enumerate(data_list):
        question = item['question']
        answer = item['answer']
        answer_char_count_total += len(answer)

        # Formatting the question based on the model type
        if 'mistral' in model_name:
            formatted_question = f"[INST] {question} [/INST]"
        else:
            formatted_question = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
        
        q_and_a = formatted_question + ' ' + answer
        answer_inputs = tokenizer(answer, return_tensors="pt")
        answer_input_ids = answer_inputs['input_ids']
        answer_input_ids = answer_input_ids.to(device)

        # Decoding each token ID into its corresponding token
        tokens = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in answer_input_ids[0]]
        token_len = len(tokens)
        token_len_list.append(token_len)
        # print(tokens)

        q_and_a_inputs = tokenizer(q_and_a, return_tensors="pt")
        q_a_input_ids = q_and_a_inputs["input_ids"].to(device)
        question_inputs = tokenizer(formatted_question, return_tensors="pt")
        question_len = question_inputs["input_ids"].shape[1]
        labels = q_a_input_ids.clone()
        labels[:, :question_len] = -100

        with torch.no_grad():
            outputs = model(input_ids=q_a_input_ids, labels=labels)
            loss = outputs.loss
            perplexity = torch.exp(loss)
            loss_list.append(loss.item())

            # Forward pass: answer only (to get perplexity of answer alone)
            # Here, we compute loss on answer_input_ids starting from the second token
            
            if len(answer_input_ids[0]) > 1:
                outputs_y_only = model(input_ids=answer_input_ids, labels=answer_input_ids)
                y_only_loss = outputs_y_only.loss
                y_only_perplexity = torch.exp(y_only_loss)

                # Compute IDF
                IDF = perplexity / y_only_perplexity
                IDF_list.append(IDF.item())

            if perplexity > 30:
                # a = 1
                if not_cap_perplexity:
                    perplexity_list.append(perplexity.item())
            else:
                perplexity_list.append(perplexity.item())
            
            if perplexity_list == []:
                perplexity_list.append(1000000)

    average_perplexity = sum(perplexity_list) / len(perplexity_list) if perplexity_list else float('inf')
    return average_perplexity


from torch.nn.utils.rnn import pad_sequence

def in_context_perplexity_plus_calculation(data_list, model, tokenizer, model_name, cutoff=3, hidden_context="some_hidden_context", device='cuda'):
    model.to(device)
    model.eval()  # 评估模式

    perplexity_list = []
    
    # 对隐藏上下文先进行编码，注意这里假设它只会产生一个token（如果产生多个，可以选择第一个或按需处理）
    hidden_context_ids = tokenizer.encode(hidden_context, add_special_tokens=False, return_tensors='pt').to(device)
    # 如果 hidden_context_ids 长度大于1，则取第一个 token 的 id
    if hidden_context_ids.shape[1] > 1:
        hidden_context_ids = hidden_context_ids[:, :1]
    
    for item in data_list:
        question = item['question']
        answer = item['answer']
        
        # 根据模型类型构造 prompt
        if 'mistral' in model_name:
            formatted_question = f"[INST] {question} [/INST]"
        else:
            formatted_question = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
        
        # 对问题和答案分别进行编码
        question_ids = tokenizer.encode(formatted_question, return_tensors='pt').to(device)[0]
        answer_ids = tokenizer.encode(answer, return_tensors='pt').to(device)[0]
        
        neg_log_probs = []  # 用于累计每个 token 的负 log 概率
        
        # 遍历答案中的每个 token
        for i in range(len(answer_ids)):
            # 根据当前 token 的索引构造上下文
            if i < cutoff:
                # 小于 cutoff 时，直接使用问题和之前全部答案 token
                if i == 0:
                    context_ids = question_ids  # 没有答案 token
                else:
                    context_ids = torch.cat([question_ids, answer_ids[:i]])
            else:
                # 当 i >= cutoff 时，构造的上下文：问题 + hidden_context + 最近 (cutoff - 1) 个答案 token
                context_part = answer_ids[i - (cutoff - 1): i]
                context_ids = torch.cat([question_ids, hidden_context_ids.squeeze(0), context_part])
            
            # 计算当前上下文下预测下一个 token 的概率  
            # 模型在自回归生成时：给定 context_ids 得到 logits，最后一个位置预测下一个 token
            inputs = context_ids.unsqueeze(0)  # 增加 batch 维度
            with torch.no_grad():
                outputs = model(input_ids=inputs)
            logits = outputs.logits  # [1, seq_len, vocab_size]
            
            # 取最后一个位置（预测下一个 token）的 logits，然后计算 softmax 后的 log 概率
            logits_last = logits[0, -1, :]
            log_probs = torch.log_softmax(logits_last, dim=-1)
            
            target_token_id = answer_ids[i]
            token_log_prob = log_probs[target_token_id]
            neg_log_probs.append(-token_log_prob)
        
        # 计算当前答案的平均负 log 概率，再 exponentiate 得到 perplexity
        if len(neg_log_probs) > 0:
            avg_neg_log_prob = sum(neg_log_probs) / len(neg_log_probs)
            example_ppl = torch.exp(avg_neg_log_prob)
        else:
            example_ppl = torch.tensor(float('inf'))
        
        perplexity_list.append(example_ppl.item())
    
    average_perplexity = sum(perplexity_list) / len(perplexity_list) if perplexity_list else float('inf')
    return average_perplexity





def compute_kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    kl_per_token = p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))
    kl_per_token = kl_per_token.sum(dim=-1)
    kl_mean = kl_per_token.mean()
    return kl_mean.item()

def kl_calculation(
    data_list, 
    model, 
    tokenizer, 
    model_name, 
    initial_prediction_list,  # <--- 新增参数
    device='cuda'
):
    loss_list = []
    kl_values = []  # <--- 新增，用于存放 KL 值

    for problem_id, item in enumerate(data_list):
        question = item['question']
        original_question = item['original_question']
        answer = item['answer']
        initial_prediction = initial_prediction_list[problem_id]

        if 'mistral' in model_name:
            formatted_question = f"[INST] {question} [/INST]"
            original_question = f"[INST] {original_question} [/INST]"
        else:
            formatted_question = (
                f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
            )

        # ground truth 的文本拼接
        q_and_a_gt = formatted_question + ' ' + answer
        
        # initial prediction 的文本拼接
        q_and_a_pred = formatted_question + ' ' + initial_prediction
        
        # 分别 token 化
        q_and_a_gt_inputs = tokenizer(q_and_a_gt, return_tensors="pt").to(device)
        q_and_a_pred_inputs = tokenizer(q_and_a_pred, return_tensors="pt").to(device)

        # 计算 ground truth 的 labels
        question_inputs = tokenizer(formatted_question, return_tensors="pt").to(device)
        question_len = question_inputs["input_ids"].shape[1]
        
        # 构造 labels，让问题部分不计算 loss
        labels_gt = q_and_a_gt_inputs["input_ids"].clone()
        labels_gt[:, :question_len] = -100
        
        with torch.no_grad():
            # ground truth forward
            outputs_gt = model(
                input_ids=q_and_a_gt_inputs["input_ids"],
                labels=labels_gt
            )
            loss_gt = outputs_gt.loss
            loss_list.append(loss_gt.item())

            # initial prediction forward（不一定需要加 labels，这里示例把它当做“仅获取 logits”）
            outputs_pred = model(
                input_ids=q_and_a_pred_inputs["input_ids"],
                labels=None  # 或者你也可以构造对应的 labels，这里要看你的需求
            )
            
            # 计算 KL
            # 注意：由于计算 KL 需要两者的 logits 维度一致，你可能需要让 ground truth 
            # 和 initial prediction 的序列长度对齐，这里仅演示思路，具体实现需根据自己的数据处理逻辑。
            # 如果两者长度不同，需要在 seq_len 维度上做截断或 padding。
            logits_p = outputs_gt.logits  # shape: [batch_size, seq_len, vocab_size]
            logits_q = outputs_pred.logits
            
            # 在这里根据自己的需求，对齐两者的长度
            min_len = min(logits_p.shape[1], logits_q.shape[1])
            logits_p = logits_p[:, :min_len, :]
            logits_q = logits_q[:, :min_len, :]

            kl_value = compute_kl_divergence(logits_p, logits_q)
            kl_values.append(kl_value)

    average_loss = sum(loss_list) / len(loss_list) if loss_list else float('inf')
    average_kl = sum(kl_values) / len(kl_values) if kl_values else float('inf')

    return average_loss, average_kl




def kl_calculation_ablation_study(
    data_list, 
    model, 
    tokenizer, 
    model_name, 
    device='cuda'
):
    loss_list = []
    kl_values = []  # <--- 新增，用于存放 KL 值

    for problem_id, item in enumerate(data_list):
        question = item['question']
        original_question = item['original_question']
        answer = item['answer']

        if 'mistral' in model_name:
            formatted_question = f"[INST] {question} [/INST]"
            original_question = f"[INST] {original_question} [/INST]"
        else:
            formatted_question = (
                f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
            )

        # ground truth 的文本拼接
        q_and_a_gt = formatted_question + ' ' + answer
        
        # initial prediction 的文本拼接
        
        # 分别 token 化
        q_and_a_gt_inputs = tokenizer(q_and_a_gt, return_tensors="pt").to(device)

        # 计算 ground truth 的 labels
        question_inputs = tokenizer(formatted_question, return_tensors="pt").to(device)
        question_len = question_inputs["input_ids"].shape[1]
        
        # 构造 labels，让问题部分不计算 loss
        labels_gt = q_and_a_gt_inputs["input_ids"].clone()
        labels_gt[:, :question_len] = -100
        
        with torch.no_grad():
            # ground truth forward
            outputs_gt = model(
                input_ids=q_and_a_gt_inputs["input_ids"],
                labels=labels_gt
            )
            loss_gt = outputs_gt.loss
            loss_list.append(loss_gt.item())

            # 计算 KL
            # 注意：由于计算 KL 需要两者的 logits 维度一致，你可能需要让 ground truth 
            # 和 initial prediction 的序列长度对齐，这里仅演示思路，具体实现需根据自己的数据处理逻辑。
            # 如果两者长度不同，需要在 seq_len 维度上做截断或 padding。
            # logits_p = outputs_gt.logits  # shape: [batch_size, seq_len, vocab_size]
            
            # 在这里根据自己的需求，对齐两者的长度
            # min_len = min(logits_p.shape[1], logits_q.shape[1])
            # logits_p = logits_p[:, :, :]
            # logits_q = logits_q[:, :min_len, :]

            # kl_value = compute_kl_divergence(logits_p, logits_q)
            # kl_values.append(kl_value)

    average_loss = sum(loss_list) / len(loss_list) if loss_list else float('inf')
    # average_kl = sum(kl_values) / len(kl_values) if kl_values else float('inf')

    return average_loss#, average_kl