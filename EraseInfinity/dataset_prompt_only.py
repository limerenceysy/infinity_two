# coding: UTF-8
"""
    @date:  2025.01
    @func:  Prompt-only Dataset for EraseInfinity
            只使用文本prompt的数据集，不需要真实nude图像
            基于ESD loss的原理：训练只需要prompt，pixel_values只用于提供shape
"""

import os
import random
from typing import Optional, List

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

# 设置 NLTK 数据路径，使用项目目录下的 wordnet
# 与 dataset.py 保持一致
import nltk
script_dir = os.path.dirname(os.path.abspath(__file__))
nltk_data_dir = os.path.join(script_dir, 'nltk_data')
if os.path.exists(nltk_data_dir):
    nltk.data.path.insert(0, nltk_data_dir)  # 优先使用项目目录下的 wordnet


def get_synonyms(word: str) -> set:
    """
    获取单词的同义词
    """
    try:
        import nltk
        from nltk.corpus import wordnet
        
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name().replace('_', ' '))
        return set(synonyms) if synonyms else {word}
    except Exception as e:
        # 如果NLTK不可用，返回预定义的nude同义词列表
        nude_synonyms = {
            'nude', 'naked', 'bare', 'unclothed', 'undressed', 
            'exposed', 'uncovered', 'stripped', 'au naturel'
        }
        if word.lower() in ['nude', 'naked', 'nudity']:
            return nude_synonyms
        return {word}


class PromptOnlyDataset(Dataset):
    """
    只使用prompt的数据集，不需要真实nude图像
    
    核心原理：
    - ESD loss只需要text prompt来生成中间状态z
    - pixel_values只用来提供batch size和shape信息
    - 因此可以使用随机噪声或固定图像作为pixel_values
    """
    
    def __init__(
        self,
        instance_prompt: str,
        key_word: str,
        size: int = 256,
        num_samples: int = 100,  # 虚拟样本数量
        use_random_noise: bool = True,  # True=使用随机噪声，False=使用固定灰色图像
    ):
        """
        Args:
            instance_prompt: 要擦除的概念的prompt（如"nude person"）
            key_word: 核心关键词（如"nude"）
            size: 图像大小
            num_samples: 虚拟样本数量（用于模拟epoch）
            use_random_noise: 是否使用随机噪声（否则使用固定灰色图像）
        """
        
        self.size = size
        self.num_samples = num_samples
        self.use_random_noise = use_random_noise
        self.instance_prompt = instance_prompt
        self.key_word = key_word
        
        print(f"=" * 80)
        print(f"PromptOnlyDataset initialized:")
        print(f"  - Prompt: {instance_prompt}")
        print(f"  - Key word: {key_word}")
        print(f"  - Image size: {size}x{size}")
        print(f"  - Virtual samples: {num_samples}")
        print(f"  - Using {'random noise' if use_random_noise else 'fixed gray image'}")
        print(f"  - ⚠️  No real nude images needed!")
        print(f"=" * 80)
        
        # 如果使用固定图像，预先生成一个灰色图像
        if not use_random_noise:
            self.fixed_image = self._create_gray_image()
            print("  ✓ Fixed gray image created")
    
    def _create_gray_image(self) -> torch.Tensor:
        """创建一个固定的灰色图像（归一化到[-1, 1]）"""
        # 创建灰色图像 (0.5, 0.5, 0.5) 归一化后为 (0, 0, 0)
        image = torch.ones(3, self.size, self.size) * 0.5
        # 归一化到 [-1, 1]
        image = (image - 0.5) / 0.5
        return image
    
    def _create_random_noise(self) -> torch.Tensor:
        """创建随机噪声图像（归一化到[-1, 1]）"""
        # 生成 [0, 1] 范围的随机噪声
        noise = torch.rand(3, self.size, self.size)
        # 归一化到 [-1, 1]
        noise = (noise - 0.5) / 0.5
        return noise
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        example = {}
        
        # ==================== 生成pixel_values ====================
        # 根据ESD loss原理，pixel_values只用于提供shape信息
        # 因此可以使用随机噪声或固定图像
        if self.use_random_noise:
            pixel_values = self._create_random_noise()
        else:
            pixel_values = self.fixed_image.clone()
        
        example["pixel_values"] = pixel_values
        
        # ==================== 生成prompt ====================
        prompt = self.instance_prompt
        
        # Prompt 增强：随机打乱单词顺序（10% 概率保持原样）
        if random.random() >= 0.1:
            words = prompt.split(" ")
            random.shuffle(words)
            prompt = ' '.join(words)
        
        example["prompt"] = prompt
        
        # ==================== 获取同义词 ====================
        if self.key_word is not None:
            synonym_list = list(get_synonyms(self.key_word))
            if len(synonym_list) > 0:
                synonym = random.choice(synonym_list)
            else:
                synonym = self.key_word
            
            example["synonym"] = synonym
            
            # 50% 概率用同义词替换关键词
            if random.random() >= 0.5 and self.key_word in prompt:
                prompt = prompt.replace(self.key_word, synonym)
                example["prompt"] = prompt
        else:
            example["synonym"] = ""
        
        example["key_word"] = self.key_word if self.key_word is not None else ""
        
        return example


def collate_fn(examples: List[dict]) -> dict:
    """
    Collate function for DataLoader
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    prompts = [example["prompt"] for example in examples]
    synonyms = [example["synonym"] for example in examples]
    key_words = [example["key_word"] for example in examples]
    
    batch = {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "synonyms": synonyms,
        "key_words": key_words,
    }
    
    return batch


if __name__ == "__main__":
    # 测试数据集
    print("Testing PromptOnlyDataset...")
    print()
    
    # 测试1：使用随机噪声
    dataset_noise = PromptOnlyDataset(
        instance_prompt="nude person with exposed body",
        key_word="nude",
        size=256,
        num_samples=50,
        use_random_noise=True,
    )
    
    print(f"\nDataset size: {len(dataset_noise)}")
    
    # 测试获取样本
    sample = dataset_noise[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Pixel values shape: {sample['pixel_values'].shape}")
    print(f"Pixel values range: [{sample['pixel_values'].min():.2f}, {sample['pixel_values'].max():.2f}]")
    print(f"Prompt: {sample['prompt']}")
    print(f"Synonym: {sample['synonym']}")
    print(f"Key word: {sample['key_word']}")
    
    # 测试2：使用固定灰色图像
    print("\n" + "="*80)
    dataset_gray = PromptOnlyDataset(
        instance_prompt="nude person with exposed body",
        key_word="nude",
        size=256,
        num_samples=50,
        use_random_noise=False,
    )
    
    # 测试DataLoader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset_noise,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    print("\nTesting DataLoader...")
    for batch in dataloader:
        print(f"Batch pixel_values shape: {batch['pixel_values'].shape}")
        print(f"Batch prompts: {batch['prompts']}")
        print(f"Batch synonyms: {batch['synonyms']}")
        break
    
    print("\n" + "="*80)
    print("✓ Dataset test completed!")
    print("✓ No real nude images needed for training!")
    print("="*80)

