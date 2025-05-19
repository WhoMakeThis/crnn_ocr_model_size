import os
import random
import string
from captcha.image import ImageCaptcha

SAVE_DIR = 'dataset'
NUM_IMAGES = 50000
LABEL_LEN = 5
CHARS = string.ascii_uppercase + string.digits

os.makedirs(SAVE_DIR, exist_ok=True)
used_labels = set()
image = ImageCaptcha(width=150, height=50)

for i in range(NUM_IMAGES):
    while True:
        label = ''.join(random.choices(CHARS, k=LABEL_LEN))
        if label not in used_labels:
            used_labels.add(label)
            break
    img = image.generate_image(label)
    img.save(os.path.join(SAVE_DIR, f'{label}.png'))
    if (i+1) % 1000 == 0:
        print(f'{i+1} / {NUM_IMAGES} 생성 완료')

print(f'✅ {NUM_IMAGES}개의 캡차 이미지가 {SAVE_DIR} 폴더에 생성되었습니다.')
