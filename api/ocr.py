from paddleocr import PaddleOCR,draw_ocr
ocr = PaddleOCR(lang='en') # need to run only once to download and load model into memory
from PIL import Image

img1= Image.open(r"E:\git\Door_number_finding_system\api\images\second.jpg")
h,w=img1.size
img1.save(r"E:\git\Door_number_finding_system\api\original.png")

img_path = r"E:\git\Door_number_finding_system\api\original.png"

result = ocr.ocr(img_path, cls=False)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

# draw result

result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path=r'C:\Windows\Fonts/arial.ttf')
im_show = Image.fromarray(im_show)
im_show.save(r"E:\git\Door_number_finding_system\api\result.jpg")