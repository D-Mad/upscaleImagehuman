import gradio as gr
import cv2
from PIL import Image, ImageOps
import os
from pathlib import Path
import dlib
from datetime import datetime
# Khởi tạo phát hiện khuôn mặt và dự đoán điểm mốc từ dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Cần tải file này

def process_single_image(input_image_path, scale_factor, output_width, output_height):
    # Load ảnh
    image = cv2.imread(input_image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = detector(gray_image)

    if len(faces) == 0:
        return "Không tìm thấy khuôn mặt nào trong ảnh."

    # Lấy khuôn mặt đầu tiên và điểm mốc
    face = faces[0]
    landmarks = predictor(gray_image, face)

    # Tính trung tâm khuôn mặt dựa trên điểm mốc (ví dụ: sử dụng điểm giữa của hai mắt)
    center_x = (landmarks.part(36).x + landmarks.part(45).x) // 2
    center_y = (landmarks.part(36).y + landmarks.part(45).y) // 2

    # Mở ảnh bằng PIL và scale
    pil_image = Image.open(input_image_path)
    scaled_image = pil_image.resize((int(pil_image.width * scale_factor), int(pil_image.height * scale_factor)), Image.Resampling.LANCZOS)

    # Điều chỉnh vị trí cắt dựa trên trung tâm khuôn mặt
    left = max(center_x * scale_factor - output_width // 2, 0)
    top = max(center_y * scale_factor - output_height // 2, 0)
    right = min(left + output_width, scaled_image.width)
    bottom = min(top + output_height, scaled_image.height)

    # Cắt và điều chỉnh kích thước ảnh cuối cùng
    final_image = scaled_image.crop((left, top, right, bottom))

    return final_image
def process_images_in_folder(input_folder_path, scale_factor, output_width, output_height, output_folder_path):
    # Kiểm tra và tạo thư mục đầu ra nếu cần
    Path(output_folder_path).mkdir(parents=True, exist_ok=True)
    messages = []

    for root, dirs, files in os.walk(input_folder_path):
        # Bỏ qua thư mục đầu ra khi duyệt file
        if root.startswith(output_folder_path):
            continue

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_image_path = os.path.join(root, file)
                # Tạo đường dẫn thư mục đầu ra tương ứng
                relative_path = os.path.relpath(root, input_folder_path)
                output_dir = os.path.join(output_folder_path, relative_path)
                Path(output_dir).mkdir(parents=True, exist_ok=True)

                try:
                    # Xử lý ảnh
                    output_image_path = os.path.join(output_dir, file)
                    final_image = process_single_image(input_image_path, scale_factor, output_width, output_height)
                    final_image.save(output_image_path)
                    messages.append(f"Xử lý thành công ảnh: {os.path.join(relative_path, file)}")
                except Exception as e:
                    messages.append(f"Không thể xử lý ảnh: {os.path.join(relative_path, file)}. Lỗi: {str(e)}")

    return "\n".join(messages)


def preview_scaled_image(input_image, scale_factor, output_width, output_height):
    # Sử dụng hàm process_single_image để lấy ảnh đã scale
    scaled_image = process_single_image(input_image.name, scale_factor, output_width, output_height)
    
    # Trả về ảnh đã scale để hiển thị như một xem trước
    return scaled_image

def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## Chọn thư mục và scale nhiều ảnh")
        with gr.Row():
            input_folder = gr.Textbox(label="Nhập đường dẫn thư mục đầu vào")
            scale_factor = gr.Number(label="Tỷ lệ scale", value=1.5, step=0.1)
            output_width = gr.Number(label="Chiều rộng đầu ra", value=2160, step=10)
            output_height = gr.Number(label="Chiều cao đầu ra", value=3840, step=10)
            submit_button = gr.Button("Xử lý tất cả ảnh trong thư mục")
        with gr.Row():
            gr.Markdown("## Xem trước ảnh")
            with gr.Column():
                preview_input = gr.File(label="Chọn ảnh để xem trước")
                preview_button = gr.Button("Xem trước")
            with gr.Column():
                preview_output = gr.Image(label="Xem trước ảnh đã scale")

        # Cài đặt hàm callback cho nút submit để xử lý ảnh
        def handle_submit(input_folder_path, scale_factor, output_width, output_height):
            # Tạo tên thư mục đầu ra dựa trên thời gian hiện tại
            output_folder_name = datetime.now().strftime("output_%Y%m%d_%H%M%S")
            # Tạo đường dẫn thư mục đầu ra bên trong thư mục đầu vào
            output_folder_path = os.path.join(input_folder_path, output_folder_name)
            
            # Kiểm tra và tạo thư mục đầu ra nếu cần
            Path(output_folder_path).mkdir(parents=True, exist_ok=True)
            
            # Gọi hàm xử lý ảnh với đường dẫn đầu ra đã được tạo
            messages = process_images_in_folder(input_folder_path, scale_factor, output_width, output_height, output_folder_path)
            return messages


        # Liên kết nút submit với hàm handle_submit
        submit_button.click(
            handle_submit,
            inputs=[input_folder, scale_factor, output_width, output_height],
            outputs=[]
        )

        preview_button.click(
            preview_scaled_image,
            inputs=[preview_input, scale_factor, output_width, output_height],
            outputs=[preview_output]
        )


    demo.launch()




gradio_interface()

