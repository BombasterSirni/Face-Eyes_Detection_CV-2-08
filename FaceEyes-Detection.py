import cv2
import numpy as np
from typing import Tuple
import os
import argparse
from sklearn.datasets import fetch_lfw_people
from mtcnn import MTCNN


def readAndPreprocessing(img_array: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Предобрабатывает входной массив изображения для обнаружения лиц и глаз.
    Нормализует значения пикселей, преобразует серое изображение в BGR при необходимости
    и изменяет размер изображения.
    
    Args:
        img_array (np.ndarray): Входной массив изображения (серое или нормализованное [0, 1]).
        size (Tuple[int, int]): Целевой размер (ширина, высота) для изменения размера.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Обработанное цветное изображение и его серое представление.
        
    Raises:
        ValueError: Если размер изображения некорректен или тип данных не поддерживается.
    """
    
    if not isinstance(img_array, np.ndarray):
        raise ValueError("Входной массив должен быть numpy.ndarray.")
    
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
            
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        
        
    try:
        img = cv2.resize(img_array, size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    except cv2.error as e:
        raise ValueError(f"Ошибка при изменении размера или преобразовании изображения: {e}")
    
    
    return img, gray


def nms(boxes: np.ndarray, iou: float = 0.5) -> np.ndarray:
    """
    Применяет подавление немаксимумов (NMS) для фильтрации перекрывающихся прямоугольников.
    Сохраняет прямоугольник с наибольшей площадью и удаляет остальные с IoU выше порога

    Args:
        boxes (np.ndarray): Массив прямоугольников [x, y, w, h].
        iou (float, optional): Порог пересечения по объединению. По умолчанию 0.5.

    Returns:
        np.ndarray: Отфильтрованный массив неперекрывающихся прямоугольников.
    """
    
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(areas)[::-1]
    pick = []

    while len(indices) > 0:
        i = indices[0]
        pick.append(i)
        indices = indices[1:]
        if len(indices) == 0:
            break

        # Вычисляем перекрытие между текущим прямоугольником и остальными
        xx1 = np.maximum(x1[i], x1[indices])
        yy1 = np.maximum(y1[i], y1[indices])
        xx2 = np.minimum(x2[i], x2[indices])
        yy2 = np.minimum(y2[i], y2[indices])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[indices]
        indices = indices[overlap < iou]

    return boxes[pick].astype(np.int32)


def main():
    """Основная функция для выполнения обнаружения лиц и глаз на наборе данных LFW.

    Парсит аргументы командной строки, обрабатывает изображения, обнаруживает лица с помощью MTCNN,
    а глаза — с использованием ключевых точек MTCNN или каскадов Haar в зависимости от --mtcnn-eye.
    Сохраняет и при необходимости отображает результаты.
    
    Raises:
        FileNotFoundError: Если файл каскада Haar отсутствует.
        ValueError: Если аргументы некорректны или данные изображения повреждены.
        RuntimeError: Если MTCNN не может выполнить обнаружение.
    """
    
    parser = argparse.ArgumentParser(description="Улучшенное обнаружение лиц и глаз с использованием MTCNN и набора данных LFW.")
    parser.add_argument('--max_images', type=int, default=5, 
                        help='Максимальное количество обрабатываемых изображений')
    
    parser.add_argument('--export_image_folder', type=str, default='export', 
                        help='Папка для сохранения обработанных изображений')
    
    parser.add_argument('--resize', type=float, default=1.0, 
                        help='Коэффициент изменения размера изображений LFW')
    
    parser.add_argument('--scaleFactor_eye', type=float, default=1.07, 
                        help='Коэффициент масштабирования для каскадов Haar')
    
    parser.add_argument('--minNeighbors_eye', type=int, default=5, 
                        help='Минимальное количество соседей для каскадов Haar')
    
    parser.add_argument('--iou_threshold', type=float, default=0.3, 
                        help='Порог IoU для NMS')
    
    parser.add_argument('--min_eye_size', type=int, default=15, 
                        help='Минимальный размер глаз в пикселях')
    
    parser.add_argument('--show', action='store_true', 
                        help='Показывать обработанные изображения')
    
    parser.add_argument('--mtcnn-eye', action='store_true', 
                        help='Использовать ключевые точки MTCNN для обнаружения глаз вместо каскадов Haar')
    
    args = parser.parse_args()

    if args.max_images <= 0:
        raise ValueError("Максимальное количество изображений должно быть положительным.")


    os.makedirs(args.export_image_folder, exist_ok=True)

    lfw_people = fetch_lfw_people(min_faces_per_person=0, resize=args.resize)
    images = lfw_people.images
    n_samples = images.shape[0]
    max_images = min(args.max_images, n_samples)
    selected_images = images[:max_images]

    # Инициализация MTCNN для обнаружения лиц
    try:
        detector = MTCNN()
    except Exception as e:
        raise RuntimeError(f"Ошибка инициализации MTCNN: {e}")
    
    # Относительный путь к каскаду для глаз (используется только если mtcnn-eye=False)
    cascade_dir = os.path.dirname(__file__)  # Путь к директории скрипта
    eye_xml = os.path.join(cascade_dir, "haarcascade_eye_tree_eyeglasses.xml")
    
    if not args.mtcnn_eye and not os.path.exists(eye_xml):
        raise FileNotFoundError(f"Файл каскада {eye_xml} не найден. Поместите его в папку проекта.")
    
    eye_cascade = cv2.CascadeClassifier(eye_xml) if not args.mtcnn_eye else None

    for i, image in enumerate(selected_images):
        
        try:
            img, img_gray = readAndPreprocessing(image, size=(500, 500))
        except ValueError as e:
            print(f"[ОШИБКА] Не удалось обработать изображение {i}: {e}")
            continue
        
        
        # Обнаружение лиц с помощью MTCNN
        try:
            detections = detector.detect_faces(img)
        except Exception as e:
            print(f"[ОШИБКА] Не удалось обнаружить лица на изображении {i}: {e}")
            detections = []
        
        
        # Извлечение bounding boxes (конвертация в np.array [x, y, w, h])
        faces = np.array([det['box'] for det in detections]) if detections else np.empty((0, 4), dtype=np.int32)
        faces = nms(faces, iou=args.iou_threshold)

        for (x, y, w, h) in faces:
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            if args.mtcnn_eye:
                
                # Обнаружение глаз с использованием ключевых точек MTCNN
                for det in detections:
                    if det['box'] == [x, y, w, h]:
                        keypoints = det['keypoints']
                        
                        left_eye = (int(keypoints['left_eye'][0] - x), int(keypoints['left_eye'][1] - y))
                        right_eye = (int(keypoints['right_eye'][0] - x), int(keypoints['right_eye'][1] - y))
                        eye_w, eye_h = 80, 80  # Размер прямоугольника для глаз
                        eyes = [
                            [left_eye[0] - eye_w//2, left_eye[1] - eye_h//2, eye_w, eye_h],
                            [right_eye[0] - eye_w//2, right_eye[1] - eye_h//2, eye_w, eye_h]
                        ]
                        break
            else:
                
                # Обнаружение глаз с использованием каскадов Haar
                if eye_cascade is not None:
                    eyes = eye_cascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=args.scaleFactor_eye,
                        minNeighbors=args.minNeighbors_eye,
                        minSize=(args.min_eye_size, args.min_eye_size)
                    )
                    if len(eyes) > 0:
                        eyes = nms(eyes, iou=args.iou_threshold)
                        # Фильтрация глаз: должны быть в верхней половине лица и на одной высоте
                        eyes = [e for e in eyes if e[1] < h * 0.55]
                        if len(eyes) >= 2:
                            mean_y = np.mean([ey for (_, ey, _, _) in eyes])
                            eyes = [e for e in eyes if abs(e[1] - mean_y) < h * 0.15]


            # Отрисовка результатов
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Красный прямоугольник для лиц
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Зелёный прямоугольник для глаз


        # Сохранение готовых изображений в папку
        try:
            export_path = os.path.join(args.export_image_folder, f"processed_lfw_{i}.jpg")
            cv2.imwrite(export_path, img)
            print(f"[ИНФО] Сохранено {export_path} ({len(faces)} лиц обнаружено)")
        except Exception as e:
            print(f"[ОШИБКА] Не удалось сохранить изображение {i}: {e}")



        if args.show:
            cv2.imshow("Результат", img)
            cv2.waitKey(0)


    if args.show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
