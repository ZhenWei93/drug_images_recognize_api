# import os
# from glob import glob
# from PIL import Image
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from flask import Flask, request, jsonify
# import json
# import sys
# import io
# import requests
# import zipfile
# import logging


# # 安裝必要的套件（請先在終端執行這些命令）
# # pip install sentence-transformers
# # pip install faiss-cpu
# # pip install requests

# # 設定日誌
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # 設定 UTF-8 編碼以顯示中文
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# # # 確保工作目錄為專案根目錄
# # os.chdir(os.path.dirname(os.path.abspath(__file__)))

# # 初始化 Flask 應用
# app = Flask(__name__)

# # 下載並解壓檔案
# def download_and_extract_zip(url, output_zip="data.zip", extract_to="."):
#     print("正在下載檔案...")
#     response = requests.get(url, stream=True)
#     if response.status_code == 200:
#         with open(output_zip, "wb") as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         print("下載完成，開始解壓...")
#         with zipfile.ZipFile(output_zip, 'r') as zip_ref:
#             zip_ref.extractall(extract_to)
#         os.remove(output_zip)  # 移除臨時 zip 檔案
#         print("解壓完成")
#     else:
#         raise Exception(f"下載失敗，狀態碼: {response.status_code}")

# # # 下載資料
# # url = "https://drive.usercontent.google.com/download?id=1rVmGMzxnxs5twJ27A4TeP1TDZb4x3pOF&export=download&authuser=0"
# # download_and_extract_zip(url)

# # 設定資料庫圖片路徑（使用 Render 的持久化磁碟）
# DATABASE_PATH = os.path.join(os.getcwd(), "medicine_images")  # 資料庫圖片資料夾
# OUTPUT_INDEX_PATH = os.path.join(os.getcwd(), "vector.index")
# QUERY_PATH = os.path.join(os.getcwd(), "query_images")  # 查詢圖片資料夾

# # 下載圖片
# if not os.path.exists(DATABASE_PATH) or not glob(os.path.join(DATABASE_PATH, "*.jpg")):
#     os.makedirs(DATABASE_PATH, exist_ok=True)
#     download_and_extract_zip(
#         "https://drive.usercontent.google.com/download?id=1rVmGMzxnxs5twJ27A4TeP1TDZb4x3pOF&export=download&authuser=0",
#         extract_to=DATABASE_PATH
#     )

# # 如果查詢資料夾不存在，創建一個並放入範例查詢圖片（這裡假設手動準備）
# if not os.path.exists(QUERY_PATH):
#     os.makedirs(QUERY_PATH)
#     print(f"已創建查詢資料夾: {QUERY_PATH}，請將查詢圖片放入此資料夾")

# # 獲取資料庫圖片檔案（不包含查詢圖片）
# database_files = glob(os.path.join(DATABASE_PATH, '*.jpg'))
# print(f"資料庫中找到 {len(database_files)} 張圖片：{[os.path.basename(f) for f in database_files]}")

# # # 顯示資料庫中的圖片
# # plt.figure(figsize=(20, max(2, len(database_files) // 5 + 1) * 4))
# # for i, image_path in enumerate(database_files):
# #     img = Image.open(image_path)
# #     plt.subplot((len(database_files) + 4) // 5, 5, i + 1)
# #     plt.imshow(img)
# #     plt.title(os.path.basename(image_path))
# #     plt.axis('off')
# # plt.tight_layout()
# # plt.show()

# # 定義生成 CLIP 嵌入的函數
# def generate_clip_embeddings(images_path, model):
#     image_paths = glob(os.path.join(images_path, '*.jpg'))  # 僅頂層
#     if not image_paths:
#         raise Exception(f"No images found in {images_path}")
#     embeddings = []
#     file_names = []
    
#     print(f"正在為資料庫生成嵌入，找到 {len(image_paths)} 張圖片")
#     for img_path in image_paths:
#         try:
#             image = Image.open(img_path).convert('RGB')  # 確保 RGB 格式
#             embedding = model.encode(image, show_progress_bar=False)
#             embeddings.append(embedding)
#             file_names.append(os.path.basename(img_path))
#             print(f"已處理: {os.path.basename(img_path)}")
#         except Exception as e:
#             print(f"處理 {img_path} 時出錯: {e}")
#             continue  # 跳過錯誤圖片
    
#     if not embeddings:
#         raise Exception("No valid embeddings generated")
#     return embeddings, image_paths, file_names

# # 設定模型
# # IMAGES_PATH = os.path.join(os.getcwd(), "medicine_images")
# # model = SentenceTransformer('clip-ViT-B-32')
# # embeddings, image_paths, file_names = generate_clip_embeddings(DATABASE_PATH, model)

# # 創建 FAISS 索引並儲存 metadata
# def create_faiss_index(embeddings, image_paths, file_names, output_path):
#     if not embeddings:
#         raise Exception("No embeddings provided for indexing")
#     dimension = len(embeddings[0])
#     index = faiss.IndexFlatIP(dimension)
#     index = faiss.IndexIDMap(index)
    
#     vectors = np.array(embeddings).astype(np.float32)
#     index.add_with_ids(vectors, np.array(range(len(embeddings))))
    
#     faiss.write_index(index, output_path)
#     print(f"索引已保存至 {output_path}")
    
#     # 儲存 metadata，包括中文名稱
#     metadata = []
#     metadata_path = output_path + '.metadata.json'

#     # # 如果已有元數據檔案，讀取現有資料
#     # if os.path.exists(metadata_path):
#     #     with open(metadata_path, 'r', encoding='utf-8') as f:
#     #         existing_metadata = json.load(f)
#     #     existing_dict = {item["file_name"]: item for item in existing_metadata}
#     # else:
#     #     existing_dict = {}

#     # 假設你有一個外部來源的藥品資訊字典（範例）
#     medicine_info = {
#         "福善美保骨錠-1.jpg": {"medicationCode":"1AMZ08", "genericName":"Alendronate", "chineseBrandName": "福善美保骨錠","englishBrandName": "Fosamax PLUS"},
#         "福善美保骨錠-2.jpg": {"medicationCode":"1AMZ08", "genericName":"Alendronate", "chineseBrandName": "福善美保骨錠","englishBrandName": "Fosamax PLUS"},
#         "芙琳亞錠-1.jpg": {"medicationCode":"1MAC12", "genericName":"Calcium Folinate", "chineseBrandName": "芙琳亞錠","englishBrandName": "Folina"},
#         "芙琳亞錠-2.jpg": {"medicationCode":"1MAC12", "genericName":"Calcium Folinate", "chineseBrandName": "芙琳亞錠","englishBrandName": "Folina"},        
#         "達滋克膜衣錠-1.jpg": {"medicationCode":"1MBD06", "genericName":"Lamivudine/Tenofovir/Doravirine", "chineseBrandName": "達滋克膜衣錠","englishBrandName": "FDelstrigo"},
#         "達滋克膜衣錠-2.jpg": {"medicationCode":"1MBD06", "genericName":"Lamivudine/Tenofovir/Doravirine", "chineseBrandName": "達滋克膜衣錠","englishBrandName": "FDelstrigo"},
#         "敵芬尼朵糖衣錠-1.jpg": {"medicationCode":"1MAD01", "genericName":"Diphenidol HCl", "chineseBrandName": "敵芬妮朵糖衣錠","englishBrandName": "Diphenidol"},
#         "解鐵定膜衣錠-1.jpg": {"medicationCode":"1MAD07", "genericName":"Deferasirox", "chineseBrandName": "解鐵定膜衣錠","englishBrandName": "Jadenu"},
#         "解鐵定膜衣錠-2.jpg": {"medicationCode":"1MAD07", "genericName":"Deferasirox", "chineseBrandName": "解鐵定膜衣錠","englishBrandName": "Jadenu"},
#         "佩你安錠-1.jpg": {"medicationCode":"1MAC08", "genericName":"Cyproheptadine HCl", "chineseBrandName": "佩你安錠","englishBrandName": "Pilian"},
#         "佩你安錠-2.jpg": {"medicationCode":"1MAC08", "genericName":"Cyproheptadine HCl", "chineseBrandName": "佩你安錠","englishBrandName": "Pilian"},
#         "法瑪鎮膜衣錠-1.jpg": {"medicationCode":"1MAF07", "genericName":"Famotidine", "chineseBrandName": "法瑪鎮膜衣錠","englishBrandName": "Famotidine"},
#         "法瑪鎮膜衣錠-2.jpg": {"medicationCode":"1MAF07", "genericName":"Famotidine", "chineseBrandName": "法瑪鎮膜衣錠","englishBrandName": "Famotidine"},
#         "睦體康腸衣錠-1.jpg": {"medicationCode":"1AMZ07", "genericName":"Mycophenolate Sodium", "chineseBrandName": "睦體康腸衣錠","englishBrandName": "Myfortic"},
#         "睦體康腸衣錠-2.jpg": {"medicationCode":"1AMZ07", "genericName":"Mycophenolate Sodium", "chineseBrandName": "睦體康腸衣錠","englishBrandName": "Myfortic"},
#         "樂伯克錠-1.jpg": {"medicationCode":"1AMG21", "genericName":"Pramipexole", "chineseBrandName": "樂伯克錠","englishBrandName": "Mirapex"},
#         "樂伯克錠-2.jpg": {"medicationCode":"1AMG21", "genericName":"Pramipexole", "chineseBrandName": "樂伯克錠","englishBrandName": "Mirapex"},
#         "諾博戈膜衣錠-1.jpg": {"medicationCode":"1MDD09", "genericName":"Darolutamide", "chineseBrandName": "諾博戈膜衣錠","englishBrandName": "Nubeqa"},
#         "諾博戈膜衣錠-2.jpg": {"medicationCode":"1MDD09", "genericName":"Darolutamide", "chineseBrandName": "諾博戈膜衣錠","englishBrandName": "Nubeqa"}
#         # "福善美保骨錠-1.jpg": {"genericName":"", "chineseBrandName": "","englishBrandName": ""},
#         # "福善美保骨錠-1.jpg": {"genericName":"", "chineseBrandName": "","englishBrandName": ""},
#     }

#     # 更新或創建元數據
#     for img_path, file_name in zip(image_paths, file_names):
#         info = medicine_info.get(file_name, {"medicationCode": "UNKNOWN", "chineseBrandName": "未知藥品"})
#         metadata.append({
#             "file_name": file_name,
#             "full_path": img_path,
#             "additional_info": info
#         })
    
#     with open(metadata_path, 'w', encoding='utf-8') as f:
#         json.dump(metadata, f, indent=4)
    
#     print(f"索引和元數據已保存至 {output_path}")
#     return index

# # 讀取 FAISS 索引與 metadata
# def load_faiss_index(index_path):
#     try:
#         index = faiss.read_index(index_path)
#         with open(index_path + '.metadata.json', 'r', encoding='utf-8') as f:
#             metadata = json.load(f)
#         metadata_dict = {item["file_name"]: item for item in metadata}
#         print(f"索引已從 {index_path} 載入")
#         return index, metadata_dict
#     except Exception as e:
#         raise Exception(f"載入索引失敗: {e}")

# # 載入模型和索引
# model = SentenceTransformer('clip-ViT-B-32')

# # 如果索引不存在，則生成
# if not os.path.exists(OUTPUT_INDEX_PATH):
#     embeddings, image_paths, file_names = generate_clip_embeddings(DATABASE_PATH, model)
#     index = create_faiss_index(embeddings, image_paths, file_names, OUTPUT_INDEX_PATH)
#     index, metadata_dict = load_faiss_index(OUTPUT_INDEX_PATH)    # 新建完之後再載入 metadata_dict
# else:
#     index, metadata_dict = load_faiss_index(OUTPUT_INDEX_PATH)

# # 檢索相似圖片
# def retrieve_similar_images(query, model, index, metadata_dict, top_k=1):  # 只返回最相似的結果
#     # 圖片查詢
#     try:
#         if isinstance(query, str):
#             query = Image.open(query)
#         query_features = model.encode(query)
#         query_features = query_features.astype(np.float32).reshape(1, -1)
#         distances, indices = index.search(query_features, top_k)
#         retrieved_metadata = [metadata_dict[list(metadata_dict.keys())[int(idx)]] for idx in indices[0]]

#         # 從 metadata 中返回最相似的藥品的中文名稱
#         if retrieved_metadata:
#             medication_code = retrieved_metadata[0]["additional_info"]["medicationCode"]
#             chinese_name = retrieved_metadata[0]["additional_info"]["chineseBrandName"]
#             return medication_code, chinese_name
#         return None, None
#     except Exception as e:
#         print(f"檢索圖片時出錯: {e}")
#         return None, None

# # API 端點：接收圖片並返回中文名稱
# @app.route('/query_image', methods=['POST'])
# def query_image():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image provided"}), 400
    
#     file = request.files['image']
#     try:
#         query_img = Image.open(file.stream).convert('RGB')
#         medication_code, chinese_name = retrieve_similar_images(query_img, model, index, metadata_dict)
        
#         if medication_code and chinese_name:
#             return jsonify({"medicationCode": medication_code, "chineseBrandName": chinese_name})
#         return jsonify({"error": "No match found"}), 404
#     except Exception as e:
#         print(f"處理查詢圖片時出錯: {e}")
#         return jsonify({"error": "Invalid image"}), 400

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 8000))  # Render 預設端口
#     app.run(host='127.0.0.1', port=port)


# # # 可視化結果（僅支援圖片查詢）
# # def visualize_results(query, retrieved_metadata):
# #     plt.figure(figsize=(15, 5))
    
# #     plt.subplot(1, len(retrieved_metadata) + 1, 1)
# #     plt.imshow(query)
# #     plt.title("查詢圖片")
# #     plt.axis('off')
    
# #     for i, metadata in enumerate(retrieved_metadata):
# #         img_path = metadata["full_path"]
# #         info = metadata["additional_info"]
# #         if isinstance(info, dict):
# #             title = f"匹配 {i + 1}\n中文名: {info.get('chineseBrandName', '未知')}\n英文名: {info.get('englishBrandName', '無')}\n學名: {info.get('genericName', '未知')}"
# #         else:
# #             title = f"匹配 {i + 1}\n資訊: {info if info else '無'}"
# #         plt.subplot(1, len(retrieved_metadata) + 1, i + 2)
# #         plt.imshow(Image.open(img_path))
# #         plt.title(title)
# #         plt.axis('off')
    
# #     plt.tight_layout()
# #     plt.show()

# # # 測試圖片查詢
# # print("測試圖片查詢")
# # query_image = os.path.join(QUERY_PATH, "諾博戈膜衣錠-2.jpg")  
# # query, retrieved_metadata = retrieve_similar_images(query_image, model, index, metadata_dict)
# # visualize_results(query, retrieved_metadata)













# import os
# from glob import glob
# from PIL import Image
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from flask import Flask, request, jsonify
# import json
# import sys
# import io
# import requests
# import zipfile
# import logging
# from waitress import serve

# # 設定日誌
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # 設定 UTF-8 編碼
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# # 初始化 Flask 應用
# app = Flask(__name__)

# # 確保工作目錄
# try:
#     os.chdir(os.path.dirname(os.path.abspath(__file__)))
#     logger.info(f"工作目錄設為: {os.getcwd()}")
# except Exception as e:
#     logger.error(f"設定工作目錄失敗: {e}")
#     sys.exit(1)

# # 下載並解壓檔案
# def download_and_extract_zip(url, output_zip="data.zip", extract_to="."):
#     logger.info("正在下載檔案...")
#     try:
#         response = requests.get(url, stream=True, timeout=30)
#         response.raise_for_status()
#         with open(output_zip, "wb") as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         logger.info("下載完成，開始解壓...")
#         with zipfile.ZipFile(output_zip, 'r') as zip_ref:
#             zip_ref.extractall(extract_to)
#         os.remove(output_zip)
#         logger.info("解壓完成")
#     except Exception as e:
#         logger.error(f"下載或解壓失敗: {e}")
#         raise

# # 設定路徑
# DATABASE_PATH = os.path.join(os.getcwd(), "medicine_images")
# OUTPUT_INDEX_PATH = os.path.join(os.getcwd(), "vector.index")
# QUERY_PATH = os.path.join(os.getcwd(), "query_images")

# # 初始化全局變數
# model = None
# index = None
# metadata_dict = None

# # 定義生成 CLIP 嵌入的函數
# def generate_clip_embeddings(images_path, model):
#     image_paths = glob(os.path.join(images_path, '*.jpg'))
#     if not image_paths:
#         raise Exception(f"No images found in {images_path}")
#     embeddings = []
#     file_names = []
    
#     logger.info(f"正在為資料庫生成嵌入，找到 {len(image_paths)} 張圖片")
#     for img_path in image_paths:
#         try:
#             image = Image.open(img_path).convert('RGB')
#             embedding = model.encode(image, show_progress_bar=False)
#             embeddings.append(embedding)
#             file_names.append(os.path.basename(img_path))
#             logger.info(f"已處理: {os.path.basename(img_path)}")
#         except Exception as e:
#             logger.error(f"處理 {img_path} 時出錯: {e}")
#             continue
    
#     if not embeddings:
#         raise Exception("No valid embeddings generated")
#     return embeddings, image_paths, file_names

# # 創建 FAISS 索引
# def create_faiss_index(embeddings, image_paths, file_names, output_path):
#     if not embeddings:
#         raise Exception("No embeddings provided for indexing")
#     dimension = len(embeddings[0])
#     index = faiss.IndexFlatIP(dimension)
#     index = faiss.IndexIDMap(index)
    
#     vectors = np.array(embeddings).astype(np.float32)
#     index.add_with_ids(vectors, np.array(range(len(embeddings))))
    
#     faiss.write_index(index, output_path)
#     logger.info(f"索引已保存至 {output_path}")
    
#     metadata = []
#     metadata_path = output_path + '.metadata.json'
#     medicine_info = {
#         "福善美保骨錠-1.jpg": {"medicationCode":"1AMZ08", "genericName":"Alendronate", "chineseBrandName": "福善美保骨錠","englishBrandName": "Fosamax PLUS"},
#         "福善美保骨錠-2.jpg": {"medicationCode":"1AMZ08", "genericName":"Alendronate", "chineseBrandName": "福善美保骨錠","englishBrandName": "Fosamax PLUS"},
#         "芙琳亞錠-1.jpg": {"medicationCode":"1MAC12", "genericName":"Calcium Folinate", "chineseBrandName": "芙琳亞錠","englishBrandName": "Folina"},
#         "芙琳亞錠-2.jpg": {"medicationCode":"1MAC12", "genericName":"Calcium Folinate", "chineseBrandName": "芙琳亞錠","englishBrandName": "Folina"},
#         "達滋克膜衣錠-1.jpg": {"medicationCode":"1MBD06", "genericName":"Lamivudine/Tenofovir/Doravirine", "chineseBrandName": "達滋克膜衣錠","englishBrandName": "FDelstrigo"},
#         "達滋克膜衣錠-2.jpg": {"medicationCode":"1MBD06", "genericName":"Lamivudine/Tenofovir/Doravirine", "chineseBrandName": "達滋克膜衣錠","englishBrandName": "FDelstrigo"},
#         "敵芬妮朵糖衣錠-1.jpg": {"medicationCode":"1MAD01", "genericName":"Diphenidol HCl", "chineseBrandName": "敵芬妮朵糖衣錠","englishBrandName": "Diphenidol"},
#         "解鐵定膜衣錠-1.jpg": {"medicationCode":"1MAD07", "genericName":"Deferasirox", "chineseBrandName": "解鐵定膜衣錠","englishBrandName": "Jadenu"},
#         "解鐵定膜衣錠-2.jpg": {"medicationCode":"1MAD07", "genericName":"Deferasirox", "chineseBrandName": "解鐵定膜衣錠","englishBrandName": "Jadenu"},
#         "佩你安錠-1.jpg": {"medicationCode":"1MAC08", "genericName":"Cyproheptadine HCl", "chineseBrandName": "佩你安錠","englishBrandName": "Pilian"},
#         "佩你安錠-2.jpg": {"medicationCode":"1MAC08", "genericName":"Cyproheptadine HCl", "chineseBrandName": "佩你安錠","englishBrandName": "Pilian"},
#         "法瑪鎮膜衣錠-1.jpg": {"medicationCode":"1MAF07", "genericName":"Famotidine", "chineseBrandName": "法瑪鎮膜衣錠","englishBrandName": "Famotidine"},
#         "法瑪鎮膜衣錠-2.jpg": {"medicationCode":"1MAF07", "genericName":"Famotidine", "chineseBrandName": "法瑪鎮膜衣錠","englishBrandName": "Famotidine"},
#         "睦體康腸衣錠-1.jpg": {"medicationCode":"1AMZ07", "genericName":"Mycophenolate Sodium", "chineseBrandName": "睦體康腸衣錠","englishBrandName": "Myfortic"},
#         "睦體康腸衣錠-2.jpg": {"medicationCode":"1AMZ07", "genericName":"Mycophenolate Sodium", "chineseBrandName": "睦體康腸衣錠","englishBrandName": "Myfortic"},
#         "樂伯克錠-1.jpg": {"medicationCode":"1AMG21", "genericName":"Pramipexole", "chineseBrandName": "樂伯克錠","englishBrandName": "Mirapex"},
#         "樂伯克錠-2.jpg": {"medicationCode":"1AMG21", "genericName":"Pramipexole", "chineseBrandName": "樂伯克錠","englishBrandName": "Mirapex"},
#         "諾博戈膜衣錠-1.jpg": {"medicationCode":"1MDD09", "genericName":"Darolutamide", "chineseBrandName": "諾博戈膜衣錠","englishBrandName": "Nubeqa"},
#         "諾博戈膜衣錠-2.jpg": {"medicationCode":"1MDD09", "genericName":"Darolutamide", "chineseBrandName": "諾博戈膜衣錠","englishBrandName": "Nubeqa"}
#     }
#     for img_path, file_name in zip(image_paths, file_names):
#         info = medicine_info.get(file_name, {"medicationCode": "UNKNOWN", "chineseBrandName": "未知藥品"})
#         metadata.append({
#             "file_name": file_name,
#             "full_path": img_path,
#             "additional_info": info
#         })
    
#     with open(metadata_path, 'w', encoding='utf-8') as f:
#         json.dump(metadata, f, indent=4)
    
#     logger.info(f"元數據已保存至 {metadata_path}")
#     return index

# # 讀取 FAISS 索引
# def load_faiss_index(index_path):
#     try:
#         index = faiss.read_index(index_path)
#         with open(index_path + '.metadata.json', 'r', encoding='utf-8') as f:
#             metadata = json.load(f)
#         metadata_dict = {item["file_name"]: item for item in metadata}
#         logger.info(f"索引已從 {index_path} 載入")
#         return index, metadata_dict
#     except Exception as e:
#         logger.error(f"載入索引失敗: {e}")
#         raise

# # 初始化模型和索引
# def initialize():
#     global model, index, metadata_dict
#     try:
#         logger.info("正在載入模型...")
#         model = SentenceTransformer('clip-ViT-B-32', device='cpu')
#         logger.info("模型載入完成")
        
#         if not os.path.exists(DATABASE_PATH) or not glob(os.path.join(DATABASE_PATH, "*.jpg")):
#             os.makedirs(DATABASE_PATH, exist_ok=True)
#             download_and_extract_zip(
#                 "https://drive.usercontent.google.com/download?id=1rVmGMzxnxs5twJ27A4TeP1TDZb4x3pOF&export=download&authuser=0",
#                 extract_to=DATABASE_PATH
#             )
        
#         if not os.path.exists(QUERY_PATH):
#             os.makedirs(QUERY_PATH)
#             logger.info(f"已創建查詢資料夾: {QUERY_PATH}")
        
#         database_files = glob(os.path.join(DATABASE_PATH, '*.jpg'))
#         logger.info(f"資料庫中找到 {len(database_files)} 張圖片：{[os.path.basename(f) for f in database_files]}")
        
#         if not os.path.exists(OUTPUT_INDEX_PATH):
#             logger.info("索引不存在，正在生成...")
#             embeddings, image_paths, file_names = generate_clip_embeddings(DATABASE_PATH, model)
#             index = create_faiss_index(embeddings, image_paths, file_names, OUTPUT_INDEX_PATH)
#             index, metadata_dict = load_faiss_index(OUTPUT_INDEX_PATH)
#         else:
#             index, metadata_dict = load_faiss_index(OUTPUT_INDEX_PATH)
#     except Exception as e:
#         logger.error(f"初始化失敗: {e}")
#         sys.exit(1)

# # 檢索相似圖片
# def retrieve_similar_images(query, model, index, metadata_dict, top_k=1):
#     try:
#         if isinstance(query, str):
#             query = Image.open(query)
#         query_features = model.encode(query)
#         query_features = query_features.astype(np.float32).reshape(1, -1)
#         distances, indices = index.search(query_features, top_k)
#         retrieved_metadata = [metadata_dict[list(metadata_dict.keys())[int(idx)]] for idx in indices[0]]
        
#         if retrieved_metadata:
#             medication_code = retrieved_metadata[0]["additional_info"]["medicationCode"]
#             chinese_name = retrieved_metadata[0]["additional_info"]["chineseBrandName"]
#             return medication_code, chinese_name
#         return None, None
#     except Exception as e:
#         logger.error(f"檢索圖片時出錯: {e}")
#         return None, None

# # 測試路由
# @app.route('/test', methods=['GET'])
# def test():
#     return jsonify({"message": "Server is running"}), 200

# # API 端點
# @app.route('/query_image', methods=['POST'])
# def query_image():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image provided"}), 400
    
#     file = request.files['image']
#     try:
#         query_img = Image.open(file.stream).convert('RGB')
#         medication_code, chinese_name = retrieve_similar_images(query_img, model, index, metadata_dict)
        
#         if medication_code and chinese_name:
#             return jsonify({"medicationCode": medication_code, "chineseBrandName": chinese_name})
#         return jsonify({"error": "No match found"}), 404
#     except Exception as e:
#         logger.error(f"處理查詢圖片時出錯: {e}")
#         return jsonify({"error": "Invalid image"}), 400

# # 初始化並啟動
# if __name__ == '__main__':
#     initialize()
#     try:
#         port = int(os.environ.get("PORT", 5000))
#         logger.info(f"啟動 Waitress 服務於 http://127.0.0.1:{port}")
#         serve(app, host='0.0.0.0', port=port, threads=4)
#     except Exception as e:
#         logger.error(f"啟動服務失敗: {e}")
#         sys.exit(1)















import os
from glob import glob
from PIL import Image
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
import json
import sys
import io
import logging
from waitress import serve

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 設定 UTF-8 編碼
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 初始化 Flask 應用
app = Flask(__name__)

# 確保工作目錄
try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"工作目錄設為: {os.getcwd()}")
except Exception as e:
    logger.error(f"設定工作目錄失敗: {e}")
    sys.exit(1)

# 設定路徑
OUTPUT_INDEX_PATH = os.path.join(os.getcwd(), "vector.index")

# 初始化全局變數
index = None
metadata_dict = None

# 讀取 FAISS 索引
def load_faiss_index(index_path):
    try:
        index = faiss.read_index(index_path)
        with open(index_path + '.metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        metadata_dict = {item["file_name"]: item for item in metadata}
        logger.info(f"索引已從 {index_path} 載入")
        return index, metadata_dict
    except Exception as e:
        logger.error(f"載入索引失敗: {e}")
        raise

# 初始化
def initialize():
    global index, metadata_dict
    try:
        if not os.path.exists(OUTPUT_INDEX_PATH):
            logger.error("索引檔案不存在，請預先生成")
            sys.exit(1)
        index, metadata_dict = load_faiss_index(OUTPUT_INDEX_PATH)
    except Exception as e:
        logger.error(f"初始化失敗: {e}")
        sys.exit(1)

# 檢索相似圖片
def retrieve_similar_images(query, metadata_dict, top_k=1):
    try:
        model = SentenceTransformer('clip-ViT-B-32', device='cpu')
        if isinstance(query, str):
            query = Image.open(query)
        query_features = model.encode(query)
        query_features = query_features.astype(np.float32).reshape(1, -1)
        distances, indices = index.search(query_features, top_k)
        retrieved_metadata = [metadata_dict[list(metadata_dict.keys())[int(idx)]] for idx in indices[0]]
        
        if retrieved_metadata:
            medication_code = retrieved_metadata[0]["additional_info"]["medicationCode"]
            chinese_name = retrieved_metadata[0]["additional_info"]["chineseBrandName"]
            return medication_code, chinese_name, model
        return None, None, model
    except Exception as e:
        logger.error(f"檢索圖片時出錯: {e}")
        return None, None, None

# 測試路由
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Server is running"}), 200

# API 端點
@app.route('/query_image', methods=['POST'])
def query_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    try:
        query_img = Image.open(file.stream).convert('RGB')
        medication_code, chinese_name, model = retrieve_similar_images(query_img, metadata_dict)
        
        if medication_code and chinese_name:
            return jsonify({"medicationCode": medication_code, "chineseBrandName": chinese_name})
        return jsonify({"error": "No match found"}), 404
    except Exception as e:
        logger.error(f"處理查詢圖片時出錯: {e}")
        return jsonify({"error": "Invalid image"}), 400
    finally:
        del model  # 釋放模型

# 初始化並啟動
if __name__ == '__main__':
    initialize()
    try:
        port = int(os.environ.get("PORT", 5000))
        logger.info(f"啟動 Waitress 服務於 http://127.0.0.1:{port}")
        serve(app, host='0.0.0.0', port=port, threads=1)
    except Exception as e:
        logger.error(f"啟動服務失敗: {e}")
        sys.exit(1)