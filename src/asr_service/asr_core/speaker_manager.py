import os
import pickle
import numpy as np
from scipy.spatial.distance import cosine
from config import REGISTERED_DB_PATH

class SpeakerManager:
    def __init__(self, threshold=0.45):
        self.threshold = threshold
        # 变更：现在存储一个向量列表，而不是单个向量
        self.teacher_embeddings = [] 
        self.teacher_name = "Teacher"
        
        self.students = [] 
        self.next_student_id = 1
        
        self.load_teacher()

    def load_teacher(self):
        """加载老师的声纹库"""
        if os.path.exists(REGISTERED_DB_PATH):
            try:
                with open(REGISTERED_DB_PATH, 'rb') as f:
                    data = pickle.load(f)
                    # 兼容旧版本：如果是单个向量，转为列表
                    emb = data.get('embedding')
                    if isinstance(emb, list):
                        self.teacher_embeddings = emb
                    else:
                        self.teacher_embeddings = [emb]
                        
                    self.teacher_name = data.get('name', 'Teacher')
                print(f"已加载老师声纹库，包含 {len(self.teacher_embeddings)} 种语气特征。")
            except Exception as e:
                print(f"加载声纹库失败: {e}")

    def save_teacher(self, name, embeddings_list):
        """保存老师声纹 (接收一个列表)"""
        # 确保列表里的每个向量都是 1D
        clean_list = []
        # 如果传入的是单个向量，先转为列表
        if not isinstance(embeddings_list, list):
            embeddings_list = [embeddings_list]
            
        for emb in embeddings_list:
            emb = np.squeeze(emb)
            if emb.ndim > 1:
                emb = emb.flatten()
            clean_list.append(emb)
            
        data = {'name': name, 'embedding': clean_list} # 保存列表
        with open(REGISTERED_DB_PATH, 'wb') as f:
            pickle.dump(data, f)
        
        self.teacher_embeddings = clean_list
        self.teacher_name = name
        print(f"老师 [{name}] 声纹已更新，共保存 {len(clean_list)} 个特征片段。")

    def identify(self, embedding):
        if embedding is None:
            return "[Unknown]"
        
        embedding = np.squeeze(embedding)
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        
        # --- 1. 比对老师 (Gallery Match) ---
        # 策略：只要与老师的【任意】一个语气特征相似度高，就是老师
        is_teacher = False
        max_teacher_score = -1
        best_teacher_idx = -1
        
        if self.teacher_embeddings:
            for idx, t_emb in enumerate(self.teacher_embeddings):
                score = 1 - cosine(embedding, t_emb)
                if score > max_teacher_score:
                    max_teacher_score = score
                    best_teacher_idx = idx
            
            # 老师的判定逻辑
            if max_teacher_score > self.threshold: 
                is_teacher = True
                # 在线更新老师声纹 (Online Learning)
                # 仅更新最匹配的那个特征向量，使其逐渐适应当前环境
                alpha = 0.15 # 学习率略低于学生，保持老师声纹的稳定性
                old_emb = self.teacher_embeddings[best_teacher_idx]
                new_emb = (1 - alpha) * old_emb + alpha * embedding
                self.teacher_embeddings[best_teacher_idx] = new_emb

        # --- 2. 比对学生 ---
        best_score = -1
        best_student_idx = -1
        
        for i, student in enumerate(self.students):
            score = 1 - cosine(embedding, student['embedding'])
            if score > best_score:
                best_score = score
                best_student_idx = i
        
        # Debug: 打印分数以便调试
        # print(f" [Debug] T:{max_teacher_score:.3f} S:{best_score:.3f} ", end="")

        debug_info = f"(T:{max_teacher_score:.2f}|S:{best_score:.2f})"

        if is_teacher:
            return f"[{self.teacher_name} {debug_info}]"
        
        if best_score > self.threshold:
            # 在线更新学生声纹
            matched_student = self.students[best_student_idx]
            alpha = 0.2
            new_emb = (1 - alpha) * matched_student['embedding'] + alpha * embedding
            self.students[best_student_idx]['embedding'] = new_emb
            return f"[{matched_student['id']} {debug_info}]"
        else:
            new_id = f"Student_{self.next_student_id}"
            self.students.append({
                'id': new_id,
                'embedding': embedding
            })
            self.next_student_id += 1
            return f"[{new_id} {debug_info}]"
